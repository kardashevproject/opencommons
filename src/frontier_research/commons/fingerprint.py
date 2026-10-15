"""Fast (<5ms) native-code structural Merkle hashing (sid) and execution-based behavioral fingerprinting (fp)."""

from __future__ import annotations

import ast
import copy
from dataclasses import dataclass
import hashlib
import inspect
import math
import time
from typing import Any, Callable


COMMUTATIVE_OPS = (ast.Add, ast.Mult, ast.BitAnd, ast.BitOr, ast.BitXor)
COMMUTATIVE_CMP = (ast.Eq, ast.NotEq)


class _AlphaNormalizer(ast.NodeTransformer):
    """Normalizes parameter and local variable names to canonical De Bruijn-like positional names ($p0, $v0)
    and strips docstrings/type annotations to compute a pure structural AST representation.
    """

    def __init__(self, func_name: str) -> None:
        super().__init__()
        self.func_name = func_name
        self.var_map: dict[str, str] = {}
        self.param_count = 0
        self.local_count = 0

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        node = copy.copy(node)
        node.name = "$self"
        node.returns = None
        node.decorator_list = []

        # Strip docstring if present
        body = list(node.body)
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            body = body[1:]

        new_args = copy.copy(node.args)
        norm_args = []
        for arg in new_args.args:
            cname = f"$p{self.param_count}"
            self.param_count += 1
            self.var_map[arg.arg] = cname
            norm_args.append(ast.arg(arg=cname, annotation=None))
        new_args.args = norm_args
        new_args.vararg = None
        new_args.kwarg = None
        node.args = new_args
        node.body = [self.visit(stmt) for stmt in body]
        return node

    def visit_Name(self, node: ast.Name) -> ast.AST:
        node = copy.copy(node)
        if node.id == self.func_name:
            node.id = "$self"
        elif isinstance(node.ctx, ast.Store):
            if node.id not in self.var_map:
                self.var_map[node.id] = f"$v{self.local_count}"
                self.local_count += 1
            node.id = self.var_map[node.id]
        elif node.id in self.var_map:
            node.id = self.var_map[node.id]
        return node

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AST:
        if node.value is None:
            return ast.Pass()
        return self.visit(ast.Assign(targets=[node.target], value=node.value))

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        node = copy.copy(node)
        node.left = self.visit(node.left)
        node.right = self.visit(node.right)
        if isinstance(node.op, COMMUTATIVE_OPS):
            d_left = ast.dump(node.left, include_attributes=False)
            d_right = ast.dump(node.right, include_attributes=False)
            if d_left > d_right:
                node.left, node.right = node.right, node.left
        return node

    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        node = copy.copy(node)
        node.left = self.visit(node.left)
        node.comparators = [self.visit(c) for c in node.comparators]
        if len(node.ops) == 1 and isinstance(node.ops[0], COMMUTATIVE_CMP):
            d_left = ast.dump(node.left, include_attributes=False)
            d_right = ast.dump(node.comparators[0], include_attributes=False)
            if d_left > d_right:
                node.left, node.comparators[0] = node.comparators[0], node.left
        return node


def compute_structural_sid(source_code: str) -> tuple[str, str, tuple[str, ...], str, tuple[str, ...]]:
    """Parse Python source code, extract primary function metadata, dependencies, and compute Merkle sid.

    Returns: (sid, func_name, param_types, return_type, called_names)
    """
    tree = ast.parse(source_code)
    func_nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if not func_nodes:
        raise ValueError("No top-level FunctionDef found in source code.")
    fn = func_nodes[-1]
    func_name = fn.name

    param_types: list[str] = []
    for a in fn.args.args:
        if a.annotation is not None:
            param_types.append(ast.unparse(a.annotation).strip())
        else:
            param_types.append("Any")

    ret_type = ast.unparse(fn.returns).strip() if fn.returns is not None else "Any"

    # Extract called identifiers (for dependency bundle closure)
    builtins_set = set(dir(__builtins__)) if isinstance(__builtins__, dict) else set(dir(__builtins__))
    called: list[str] = []
    for sub in ast.walk(fn):
        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name):
            cname = sub.func.id
            if cname != func_name and cname not in builtins_set and cname not in called:
                called.append(cname)

    norm_fn = _AlphaNormalizer(func_name).visit(copy.deepcopy(fn))
    canonical_ast = ast.dump(norm_fn, include_attributes=False)
    digest = hashlib.sha256(canonical_ast.encode("utf-8")).hexdigest()[:16]
    return f"sid:{digest}", func_name, tuple(param_types), ret_type, tuple(called)


def _canonical_serialize(val: Any) -> str:
    """Deterministically serialize arbitrary Python values (with float tolerance quantization)."""
    if val is None or isinstance(val, (bool, int, str)):
        return repr(val)
    if isinstance(val, float):
        if math.isnan(val):
            return "float:NaN"
        if math.isinf(val):
            return "float:Inf" if val > 0 else "float:-Inf"
        # Quantize to 12 significant digits (and 1e-9 near zero) so extreme-magnitude floats compare cleanly
        if abs(val) < 1e-12:
            return "float:0.000000000"
        return f"float:{val:.12g}"
    if isinstance(val, (bytes, bytearray)):
        return f"bytes:{bytes(val).hex()}"
    if isinstance(val, tuple):
        return "tuple:(" + ",".join(_canonical_serialize(x) for x in val) + ")"
    if isinstance(val, list):
        return "list:[" + ",".join(_canonical_serialize(x) for x in val) + "]"
    if isinstance(val, set):
        items = sorted(_canonical_serialize(x) for x in val)
        return "set:{" + ",".join(items) + "}"
    if isinstance(val, dict):
        items = sorted((_canonical_serialize(k), _canonical_serialize(v)) for k, v in val.items())
        return "dict:{" + ",".join(f"{k}:{v}" for k, v in items) + "}"
    if hasattr(val, "tolist"):
        return "array:" + _canonical_serialize(val.tolist())
    return f"obj:{type(val).__name__}:{repr(val)}"


def _build_probe_values_for_type(type_str: str) -> list[Any]:
    """Return 16 canonical stratified probe values for a given parameter type hint."""
    t = type_str.lower().replace(" ", "")
    if t in ("int",):
        return [0, 1, -1, 2, 3, 5, 7, 10, 12, 15, 16, 42, 100, 255, 1024, -7]
    if t in ("float",):
        return [0.0, 1.0, -1.0, 0.5, 2.0, -0.5, 0.25, 3.5, 10.0, 100.0, 0.1, 0.9, -2.5, 4.0, 8.0, 0.01]
    if t in ("bool",):
        return [False, True] * 8
    if t in ("str",):
        return [
            "",
            "a",
            "ab",
            "aba",
            "hello",
            "hello world",
            "12345",
            "A_B_C",
            "   spaces   ",
            "foo_bar_baz",
            "racecar",
            "alpha123beta",
            "test@example.com",
            "2026-10-14",
            "001100",
            "xyz",
        ]
    if t in ("bytes",):
        return [
            b"",
            b"\x00",
            b"\x01\x02",
            b"hello",
            b"\xff\x00\xff",
            b"123456789",
            b"commons",
            b"\xde\xad\xbe\xef",
            b"abc",
            b"\x00\x01\x02\x03",
            b"packet_payload",
            b"\x80\x7f",
            b"test",
            b"AAAA",
            b"\x10\x20\x30",
            b"end",
        ]
    if t in ("bytearray",):
        return [bytearray(b) for b in _build_probe_values_for_type("bytes")]
    if t.startswith("list[float]") or t.startswith("sequence[float]"):
        return [
            # 16 General Stratified Probes
            [1.0, 2.0, 3.0],
            [0.0, 0.0],
            [-1.0, 0.0, 1.0],
            [10.0, 20.0, 30.0, 40.0],
            [0.5, 1.5, 2.5],
            [3.0, 1.0, 2.0],
            [5.0],
            [2.0, 4.0, 6.0, 8.0],
            [0.1, 0.2, 0.7],
            [-5.0, -2.0, -1.0],
            [1.0, 1.0, 1.0, 1.0],
            [100.0, 1.0],
            [0.25, 0.75],
            [1.0, 4.0, 9.0, 16.0],
            [2.5, -2.5],
            [0.0, 1.0],
            # 8 Adversarial Boundary Probes
            [1000.0, 1001.0, 1002.0],
            [-1000.0, -1001.0, -1002.0],
            [1e300, 1e300],
            [0.0, 0.0, 0.0],
            [42.0],
            [],
            [1e-300, -1e-300],
            [1e15, 1.0, -1e15],
        ]
    if t.startswith("list[str]") or t.startswith("sequence[str]"):
        return [
            [],
            ["a"],
            ["a", "b", "a"],
            ["hello", "world"],
            ["foo", "bar", "baz"],
            ["x", "y", "z"],
            ["", "nonempty"],
            ["alpha", "beta"],
            ["one", "two", "two", "three"],
            ["cat", "dog"],
            ["a", "aa", "aaa"],
            ["test"],
            ["p", "q", "p", "q"],
            ["1", "2", "3"],
            ["same", "same"],
            ["first", "last"],
        ]
    if t.startswith("list[dict") or t == "list[dict]":
        return [
            [{"id": 1, "val": 10}, {"id": 2, "val": 20}],
            [{"id": 1, "val": 5}],
            [],
            [{"id": 1, "val": 10}, {"id": 1, "val": 15}],
            [{"a": 1, "b": 2}, {"a": 3, "b": 4}],
            [{"score": 0.9, "name": "x"}, {"score": 0.2, "name": "y"}],
            [{"k": "a", "n": 1}, {"k": "b", "n": 2}, {"k": "a", "n": 3}],
            [{"x": 0, "y": 0}],
        ] * 2
    if t.startswith("dict"):
        return [
            {},
            {"a": 1, "b": 2},
            {"x": 10, "y": 20, "z": 30},
            {"k1": 0, "k2": -1},
            {"alpha": 5, "beta": 5},
            {"one": 1},
            {"a": 100, "b": 50, "c": 25},
            {"key": 42},
        ] * 2
    if t.startswith("tuple"):
        return [
            (0, 0),
            (1, 2),
            (3, 4),
            (-1, 5),
            (10, 20),
            (100, 200),
            (7, 3),
            (8, 8),
        ] * 2
    # Default: list[int] or general sequence
    return [
        [],
        [1],
        [1, 2, 3],
        [3, 1, 2],
        [3, 1, 3, 2, 1],
        [0, 0, 0],
        [-3, -1, 0, 2, 5],
        [10, 20, 30, 40],
        [5, 4, 3, 2, 1],
        [2, 2, 3, 3, 3],
        [100, -100],
        [1, 3, 5, 7, 9],
        [2, 4, 6, 8],
        [42],
        [0, 1, 0, 1],
        [9, 1, 8, 2, 7],
    ]


def generate_canonical_probes(param_types: tuple[str, ...], n_probes: int = 16) -> list[tuple[Any, ...]]:
    """Construct n_probes deterministic argument tuples matching param_types."""
    if not param_types:
        return [()]
    per_param = [_build_probe_values_for_type(pt) for pt in param_types]
    probes: list[tuple[Any, ...]] = []
    for i in range(n_probes):
        args = []
        for p_idx, plist in enumerate(per_param):
            # Offset index slightly for subsequent parameters of identical type so (x, y) isn't always x == y
            offset = (i + p_idx * 3) if i >= 4 else i
            args.append(copy.deepcopy(plist[offset % len(plist)]))
        probes.append(tuple(args))
    return probes


@dataclass(frozen=True)
class FingerprintResult:
    """Result of structural + behavioral fingerprinting on a native Python function."""

    sid: str
    fp: str
    func_name: str
    param_types: tuple[str, ...]
    return_type: str
    dependencies: tuple[str, ...]
    valid_probe_count: int
    total_probe_count: int
    mutates_arguments: bool
    is_constant_output: bool
    elapsed_ms: float
    sample_outputs: tuple[str, ...]


def fingerprint_function(
    source_code: str,
    extra_globals: dict[str, Any] | None = None,
    n_probes: int = 16,
) -> FingerprintResult:
    """Compute structural Merkle sid and execution-based behavioral fingerprint fp in <5ms."""
    t0 = time.perf_counter()
    sid, func_name, param_types, return_type, deps = compute_structural_sid(source_code)

    ns: dict[str, Any] = {}
    if extra_globals:
        ns.update(extra_globals)
    exec(compile(source_code, "<commons_block>", "exec"), ns)  # noqa: S102
    fn: Callable[..., Any] = ns[func_name]

    probes = generate_canonical_probes(param_types, n_probes=n_probes)
    outcome_tokens: list[str] = []
    valid_outputs: list[str] = []
    mutates_args = False

    for probe_args in probes:
        before_serialized = [_canonical_serialize(a) for a in probe_args]
        working_args = [copy.deepcopy(a) for a in probe_args]
        try:
            res = fn(*working_args)
            after_serialized = [_canonical_serialize(a) for a in working_args]
            if after_serialized != before_serialized:
                mutates_args = True
            res_str = _canonical_serialize(res)
            state_str = "|".join(after_serialized) if mutates_args else ""
            token = f"OK:{res_str}:M({state_str})"
            outcome_tokens.append(token)
            valid_outputs.append(res_str)
        except Exception as exc:  # noqa: BLE001
            outcome_tokens.append(f"ERR:{type(exc).__name__}")

    valid_count = len(valid_outputs)
    unique_valid = len(set(valid_outputs))
    # Constant output check (if >= 4 valid probes all produce the exact same constant on varied inputs,
    # unless the return type is bool where 2 values exist)
    is_constant = (
        valid_count >= 4
        and unique_valid == 1
        and return_type.lower() not in ("bool", "none")
        and len(param_types) >= 1
    )

    if valid_count > 0:
        fp_digest = hashlib.sha256("\n".join(outcome_tokens).encode("utf-8")).hexdigest()[:16]
        fp = f"fp:{fp_digest}"
    else:
        # Fallback to structural sid if all generic probes raised domain errors
        fp = f"fp_struct:{sid[4:]}"

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return FingerprintResult(
        sid=sid,
        fp=fp,
        func_name=func_name,
        param_types=param_types,
        return_type=return_type,
        dependencies=deps,
        valid_probe_count=valid_count,
        total_probe_count=len(probes),
        mutates_arguments=mutates_args,
        is_constant_output=is_constant,
        elapsed_ms=round(elapsed_ms, 4),
        sample_outputs=tuple(valid_outputs[:4]),
    )
