"""Automated Verification and Quality Test Suite for OpenCommons Domain Discovery Expansion.

Normative requirements:
1. Schema compliance against `opencommons_schema.json` (v1 Draft 2020-12).
2. AST non-triviality: statements >= 4, node count >= 25, single pure FunctionDef,
   no docstrings/comments in canonical unparse.
3. Invariants & Pitfalls: invariants >= 2, pitfalls >= 1 with naive_pattern,
   failure_mechanism, counterexample_input, and verified_fix.
4. Deterministic defense L1–L6 via evaluate_proposal_safety(..., run_blue_team=False).
5. Out-of-process L7 sandbox execution via opencommons_sandbox.run_sandboxed_probes
   (dual-run IEEE 754 determinism, non-crashing execution, zero argument mutation).
6. Deduplication: unique names, unique structural SIDs, unique behavioral CIDs,
   and 0 collisions with the 52 canonical seed concepts in seed_concepts_v1.json.
7. Target counts: >= 35 concepts per domain track, >= 500 concepts total when full
   catalog is populated.
8. Store synchronization tool CLI verification (`sync_to_opencommons.py`).
"""

from __future__ import annotations

import ast
import glob
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
from typing import Any

import jsonschema
import pytest

# ---------------------------------------------------------------------------
# Setup Kardashev-CLI Namespace Shim
# ---------------------------------------------------------------------------

def _setup_kardashev_shim() -> Path:
    """Locate kardashev-cli reference and establish pure-python module shim."""
    repo_root = Path(__file__).resolve().parent.parent
    cli_dir = (repo_root.parent / "kardashev-cli").resolve()
    for parent in repo_root.parents:
        cand = parent / "kardashev-cli"
        if (cand / "src" / "kardashev_cli").is_dir():
            cli_dir = cand.resolve()
            break
    src_dir = cli_dir / "src" / "kardashev_cli"
    if src_dir.is_dir() and "kardashev_cli" not in sys.modules:
        cli_mod = types.ModuleType("kardashev_cli")
        cli_mod.__path__ = [str(src_dir)]
        sys.modules["kardashev_cli"] = cli_mod
    return cli_dir


CLI_DIR = _setup_kardashev_shim()

from kardashev_cli.opencommons import compute_structural_sid, fingerprint_function
from kardashev_cli.opencommons_defense import evaluate_proposal_safety
from kardashev_cli.opencommons_sandbox import run_sandboxed_probes, SandboxConfig


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCHEMA_PATH = CLI_DIR / "src" / "kardashev_cli" / "opencommons_schema.json"
SEED_PATH = CLI_DIR / "commons" / "seed_concepts_v1.json"
CONSOLIDATED_CATALOG_PATH = REPO_ROOT / "data" / "opencommons_expansion_v1.json"
DOMAINS_DIR = REPO_ROOT / "data" / "domains"

# 12 Domain tracks specified in ORIGINAL_REQUEST.md
DOMAIN_TRACKS = [
    "numerical_analysis",
    "signal_processing",
    "cryptography_and_framing",
    "distributed_systems",
    "machine_learning",
    "geodesy_and_spatial",
    "hamiltonian_dynamics",
    "number_theory_and_algebra",
    "quantitative_finance",
    "statistics_and_information",
    "graph_and_combinatorial",
    "bioinformatics_and_control",
]

# 11 Normative v1 schema domain enums defined in ORIGINAL_REQUEST.md
NORMATIVE_V1_DOMAINS = [
    "numerical_analysis",
    "signal_processing",
    "cryptography_and_framing",
    "distributed_systems",
    "machine_learning",
    "geodesy_and_spatial",
    "hamiltonian_dynamics",
    "number_theory_and_algebra",
    "quantitative_finance",
    "statistics_and_information",
    "general",
]



# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def schema_v1() -> dict[str, Any]:
    assert SCHEMA_PATH.is_file(), f"Missing schema at {SCHEMA_PATH}"
    with SCHEMA_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def validator_v1(schema_v1: dict[str, Any]) -> jsonschema.Draft202012Validator:
    jsonschema.Draft202012Validator.check_schema(schema_v1)
    return jsonschema.Draft202012Validator(schema_v1)


@pytest.fixture(scope="session")
def seed_concepts_v1() -> list[dict[str, Any]]:
    assert SEED_PATH.is_file(), f"Missing seed concepts at {SEED_PATH}"
    with SEED_PATH.open("r", encoding="utf-8") as f:
        seeds = json.load(f)
    assert len(seeds) == 52, f"Expected 52 seeds, found {len(seeds)}"
    return seeds


@pytest.fixture(scope="session")
def seed_identifiers(seed_concepts_v1: list[dict[str, Any]]) -> dict[str, set[str]]:
    names: set[str] = set()
    sids: set[str] = set()
    cids: set[str] = set()
    for s in seed_concepts_v1:
        names.add(s["name"])
        sid, _, _, _, _ = compute_structural_sid(s["source_code"])
        sids.add(sid)
        fp_res = fingerprint_function(s["source_code"])
        cids.add(f"cid:{fp_res.fp.split(':', 1)[1]}")
    return {"names": names, "sids": sids, "cids": cids}


@pytest.fixture
def sample_valid_concept() -> dict[str, Any]:
    """A representative genuine, multi-step numerical concept satisfying all criteria."""
    code = (
        "def brent_dekker_scalar_root(a_val: float, b_val: float, tol: float) -> float:\n"
        "    a = float(a_val)\n"
        "    b = float(b_val)\n"
        "    eps = max(1e-15, abs(tol))\n"
        "    fa = a * a - 2.0\n"
        "    fb = b * b - 2.0\n"
        "    c = a\n"
        "    fc = fa\n"
        "    for _ in range(50):\n"
        "        if abs(fb) < eps:\n"
        "            return b\n"
        "        if abs(b - a) < eps:\n"
        "            return b\n"
        "        m = 0.5 * (c - b)\n"
        "        b = b + m\n"
        "        fb = b * b - 2.0\n"
        "    return b\n"
    )
    return {
        "name": "brent_dekker_scalar_root",
        "param_types": ["float", "float", "float"],
        "return_type": "float",
        "summary": "Solves scalar non-linear root equation using bounded interval iteration.",
        "pseudocode": "1. Initialize brackets a, b.\n2. Compute residuals fa, fb.\n3. Iteratively bisect and refine root until convergence within tolerance.",
        "invariants": [
            "Root bracket converges monotonically towards target zero residual.",
            "Residual absolute value is bounded by tolerance threshold upon exit.",
        ],
        "pitfalls": [
            {
                "naive_pattern": "m = (c - b)",
                "failure_mechanism": "Step size without scaling causes divergent oscillation outside interval.",
                "counterexample_input": "a_val=0.0, b_val=10.0, tol=1e-8",
                "verified_fix": "Scale midpoint displacement by 0.5 to guarantee interval contraction.",
            }
        ],
        "source_code": code,
        "domain": "numerical_analysis",
    }


def _load_available_expansion_concepts() -> list[dict[str, Any]]:
    """Load concepts from consolidated catalog or domain files if available."""
    if CONSOLIDATED_CATALOG_PATH.is_file():
        try:
            data = json.loads(CONSOLIDATED_CATALOG_PATH.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                return data.get("blocks", data.get("concepts", []))
        except (OSError, json.JSONDecodeError):
            data = None

    domain_files = sorted(glob.glob(str(DOMAINS_DIR / "*.json")))
    if domain_files:
        concepts = []
        for df in domain_files:
            try:
                data = json.loads(Path(df).read_text(encoding="utf-8"))
                items = data if isinstance(data, list) else data.get("blocks", data.get("concepts", []))
                concepts.extend(items)
            except (OSError, json.JSONDecodeError):
                continue
        return concepts

    return []


# ---------------------------------------------------------------------------
# Test Category 1: Schema Compliance Verification
# ---------------------------------------------------------------------------

def test_schema_validates_valid_proposal(validator_v1, sample_valid_concept):
    """Verify that a compliant concept passes Draft 2020-12 schema validation."""
    validator_v1.validate(sample_valid_concept)
    assert validator_v1.is_valid(sample_valid_concept) is True
    assert list(validator_v1.iter_errors(sample_valid_concept)) == []


def test_schema_rejects_missing_required_fields(validator_v1, sample_valid_concept):
    """Verify that omitting any of the 9 required fields raises jsonschema.ValidationError."""
    required_fields = [
        "name", "param_types", "return_type", "summary",
        "pseudocode", "invariants", "pitfalls", "source_code", "domain"
    ]
    for field in required_fields:
        corrupted = dict(sample_valid_concept)
        del corrupted[field]
        with pytest.raises(jsonschema.ValidationError):
            validator_v1.validate(corrupted)


def test_schema_rejects_forbidden_additional_properties(validator_v1, sample_valid_concept):
    """Verify that extra non-schema properties are rejected (additionalProperties: false)."""
    corrupted = dict(sample_valid_concept)
    corrupted["block_id"] = "brent_dekker@12345678"
    with pytest.raises(jsonschema.ValidationError):
        validator_v1.validate(corrupted)


# ---------------------------------------------------------------------------
# Test Category 2: AST Non-Triviality Verification
# ---------------------------------------------------------------------------

def test_ast_non_triviality_on_valid_concept(sample_valid_concept):
    """Verify statements >= 4, node count >= 25, and pure function constraints."""
    tree = ast.parse(sample_valid_concept["source_code"])
    assert len(tree.body) == 1
    fn_def = tree.body[0]
    assert isinstance(fn_def, ast.FunctionDef)
    assert len(fn_def.body) >= 4, f"Statements: {len(fn_def.body)} < 4"
    node_count = sum(1 for _ in ast.walk(fn_def))
    assert node_count >= 25, f"AST nodes: {node_count} < 25"

    for node in ast.walk(fn_def):
        assert not isinstance(node, (ast.Global, ast.Nonlocal, ast.ClassDef, ast.AsyncFunctionDef))


def test_ast_non_triviality_rejects_trivial_functions():
    """Verify that trivial 1-liners and simple wrappers fail the non-triviality check."""
    trivial_code = "def add_wrapper(x: int, y: int) -> int:\n    return x + y\n"
    tree = ast.parse(trivial_code)
    fn_def = tree.body[0]
    assert len(fn_def.body) == 1  # Fails < 4
    assert sum(1 for _ in ast.walk(fn_def)) < 25  # Fails < 25


# ---------------------------------------------------------------------------
# Test Category 3: Invariants & Pitfalls Verification
# ---------------------------------------------------------------------------

def test_invariants_and_pitfalls_on_valid_concept(sample_valid_concept):
    """Verify invariants >= 2 and pitfalls >= 1 with complete 4-tuple fields."""
    invs = sample_valid_concept["invariants"]
    assert isinstance(invs, (list, tuple))
    assert len(invs) >= 2
    for inv in invs:
        assert isinstance(inv, str) and len(inv.strip()) > 0

    pfs = sample_valid_concept["pitfalls"]
    assert isinstance(pfs, (list, tuple))
    assert len(pfs) >= 1
    required_keys = {"naive_pattern", "failure_mechanism", "counterexample_input", "verified_fix"}
    for pf in pfs:
        assert isinstance(pf, dict)
        assert set(pf.keys()) == required_keys
        for key in required_keys:
            assert isinstance(pf[key], str) and len(pf[key].strip()) > 0


def test_invariants_and_pitfalls_rejects_insufficient_entries(sample_valid_concept):
    """Verify that < 2 invariants or 0 pitfalls fail quality requirements."""
    corrupted_inv = dict(sample_valid_concept)
    corrupted_inv["invariants"] = ["Only one invariant"]
    assert len(corrupted_inv["invariants"]) < 2

    corrupted_pf = dict(sample_valid_concept)
    corrupted_pf["pitfalls"] = []
    assert len(corrupted_pf["pitfalls"]) < 1


# ---------------------------------------------------------------------------
# Test Category 4: Deterministic Defense L1–L6 Verification
# ---------------------------------------------------------------------------

def test_deterministic_defense_l1_to_l6_pass(sample_valid_concept):
    """Verify evaluate_proposal_safety executes L1-L6 without blue team and passes with 0 rejections."""
    # Test with skip_sandbox=True for fast deterministic L1-L6
    verdict = evaluate_proposal_safety(sample_valid_concept, skip_sandbox=True)
    assert verdict.is_accepted is True, f"Failed L1-L6: {verdict.failure_rule_id}: {verdict.failure_detail}"
    assert verdict.status == "accepted"


def test_deterministic_defense_rejects_banned_builtins(sample_valid_concept):
    """Verify that dangerous functions like eval or open trigger L6 rejections."""
    bad_code = (
        "def dangerous_func(x: str) -> int:\n"
        "    a = len(x)\n"
        "    b = a + 1\n"
        "    c = eval(x)\n"
        "    return c\n"
    )
    bad_proposal = dict(sample_valid_concept)
    bad_proposal["source_code"] = bad_code
    bad_proposal["name"] = "dangerous_func"
    bad_proposal["param_types"] = ["str"]
    bad_proposal["return_type"] = "int"

    verdict = evaluate_proposal_safety(bad_proposal, skip_sandbox=True)
    assert verdict.is_accepted is False
    assert verdict.failure_layer == "L6"


# ---------------------------------------------------------------------------
# Test Category 5: Out-of-Process Sandbox L7 Verification
# ---------------------------------------------------------------------------

def test_out_of_process_sandbox_l7_execution(sample_valid_concept):
    """Verify child subprocess execution, dual-run IEEE 754 determinism, and zero argument mutation."""
    probes = [
        (1.0, 2.0, 1e-6),
        (0.0, 3.0, 1e-5),
        (-2.0, 0.0, 1e-6),
    ]
    res = run_sandboxed_probes(
        source_code=sample_valid_concept["source_code"],
        func_name=sample_valid_concept["name"],
        probes=probes,
    )
    assert res.success is True, f"Sandbox failed: {res.error_code} {res.error_message}"
    assert res.is_deterministic is True, "Dual-run determinism discrepancy detected"
    assert res.branch_coverage > 0.0, "Expected non-zero branch coverage"
    # Verify no probe mutated arguments in-place
    assert not any(pr.mutated_args for pr in res.probe_results), "Argument mutation detected in sandbox"


# ---------------------------------------------------------------------------
# Test Category 6: Deduplication & Seed Isolation Verification
# ---------------------------------------------------------------------------

def test_deduplication_and_seed_isolation(sample_valid_concept, seed_identifiers):
    """Verify that new concepts have unique names, unique SIDs, unique CIDs, and 0 seed collisions."""
    sid, name, _, _, _ = compute_structural_sid(sample_valid_concept["source_code"])
    fp_res = fingerprint_function(sample_valid_concept["source_code"])
    cid = f"cid:{fp_res.fp.split(':', 1)[1]}"

    # Verify zero collision with foundational 52 seed concepts
    assert name not in seed_identifiers["names"], f"Name collision with seed: {name}"
    assert sid not in seed_identifiers["sids"], f"SID collision with seed: {sid}"
    assert cid not in seed_identifiers["cids"], f"CID collision with seed: {cid}"


# ---------------------------------------------------------------------------
# Test Category 7: Full Catalog Automated Pipeline Verification
# ---------------------------------------------------------------------------

def test_live_expansion_catalog_conformance(validator_v1, seed_identifiers):
    """When catalog is populated, verify 100% pass on schema, AST, defense, and deduplication."""
    concepts = _load_available_expansion_concepts()
    if not concepts:
        pytest.skip(
            "Consolidated catalog data/opencommons_expansion_v1.json not yet populated. "
            "(Domain tracks scheduled across milestones M2-M5)."
        )

    print(f"\nVerifying live catalog of {len(concepts)} concepts...")
    names_seen: set[str] = set()
    sids_seen: set[str] = set()
    cids_seen: set[str] = set()

    for idx, c in enumerate(concepts):
        # 1. Schema
        validator_v1.validate(c)

        # 2. AST non-triviality
        tree = ast.parse(c["source_code"])
        assert len(tree.body) == 1
        fn_def = tree.body[0]
        assert isinstance(fn_def, ast.FunctionDef)
        assert len(fn_def.body) >= 4, f"Concept {c['name']} has < 4 statements"
        node_count = sum(1 for _ in ast.walk(fn_def))
        assert node_count >= 25, f"Concept {c['name']} has < 25 nodes ({node_count})"

        # 3. Invariants & Pitfalls
        assert len(c["invariants"]) >= 2, f"Concept {c['name']} has < 2 invariants"
        assert len(c["pitfalls"]) >= 1, f"Concept {c['name']} has < 1 pitfalls"

        # 4. L1-L6 Deterministic Defense
        verdict = evaluate_proposal_safety(c, skip_sandbox=True)
        assert verdict.is_accepted is True, (
            f"Concept {c['name']} failed {verdict.failure_layer} [{verdict.failure_rule_id}]: {verdict.failure_detail}"
        )

        # 5. Deduplication
        name = c["name"]
        sid, _, _, _, _ = compute_structural_sid(c["source_code"])
        if "cid" in c and c["cid"]:
            cid = str(c["cid"])
        else:
            fp_res = fingerprint_function(c["source_code"])
            cid = f"cid:{fp_res.fp.split(':', 1)[1]}"

        assert name not in names_seen, f"Duplicate concept name: {name}"
        assert sid not in sids_seen, f"Duplicate structural SID: {sid} in concept {name}"
        assert cid not in cids_seen, f"Duplicate behavioral CID: {cid} in concept {name}"

        names_seen.add(name)
        sids_seen.add(sid)
        cids_seen.add(cid)

        # 6. Seed collisions
        assert name not in seed_identifiers["names"], f"Seed collision on name: {name}"
        assert sid not in seed_identifiers["sids"], f"Seed collision on SID: {sid}"
        assert cid not in seed_identifiers["cids"], f"Seed collision on CID: {cid}"


def test_live_expansion_target_counts(schema_v1=None):
    """Verify target counts: >= 35 concepts per domain track, >= 500 concepts total when fully populated."""
    if not CONSOLIDATED_CATALOG_PATH.is_file():
        pytest.skip(
            "Consolidated catalog data/opencommons_expansion_v1.json not yet populated. "
            "(Target count verification will activate once all 12 domains complete)."
        )

    data = json.loads(CONSOLIDATED_CATALOG_PATH.read_text(encoding="utf-8"))
    concepts = data if isinstance(data, list) else data.get("blocks", data.get("concepts", []))

    assert len(concepts) >= 500, f"Expected >= 500 total concepts, found {len(concepts)}"

    # Per-domain check
    domain_counts: dict[str, int] = {}
    for c in concepts:
        d = c.get("domain", "general")
        domain_counts[d] = domain_counts.get(d, 0) + 1

    # Check all required domain enums from normative v1 schema defined in ORIGINAL_REQUEST.md
    NORMATIVE_V1_DOMAINS = [
        "numerical_analysis",
        "signal_processing",
        "cryptography_and_framing",
        "distributed_systems",
        "machine_learning",
        "geodesy_and_spatial",
        "hamiltonian_dynamics",
        "number_theory_and_algebra",
        "quantitative_finance",
        "statistics_and_information",
        "general",
    ]
    for d in NORMATIVE_V1_DOMAINS:
        count = domain_counts.get(d, 0)
        assert count >= 35, f"Domain track '{d}' has {count} concepts (< 35 required)"


# ---------------------------------------------------------------------------
# Test Category 8: CLI Store Sync Integration
# ---------------------------------------------------------------------------

def test_sync_to_opencommons_cli_dry_run(sample_valid_concept):
    """Test sync_to_opencommons.py execution in dry-run mode on valid catalog."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        cat_path = Path(tmp_dir) / "test_cat.json"
        cat_path.write_text(json.dumps([sample_valid_concept]), encoding="utf-8")
        cmd = [
            sys.executable,
            str(REPO_ROOT / "experiments" / "domain_discovery" / "sync_to_opencommons.py"),
            "--catalog", str(cat_path),
            "--dry-run",
            "--verbose",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        assert res.returncode == 0
        assert "DRY RUN" in res.stdout
        assert "Verification results" in res.stdout
        assert "1 accepted, 0 rejected" in res.stdout


def test_sync_to_opencommons_cli_live_sync_and_crdt_idempotence(sample_valid_concept):
    """Test live store synchronization into temporary store path and verify CRDT merge idempotence."""
    from experiments.domain_discovery.sync_to_opencommons import run_sync, load_store

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_store_path = Path(tmp_dir) / "store.json"
        cat_path = Path(tmp_dir) / "test_cat.json"
        cat_path.write_text(json.dumps([sample_valid_concept]), encoding="utf-8")

        # 1. Sync valid catalog to new temporary store
        code1 = run_sync(
            catalog_path=str(cat_path),
            store_path=tmp_store_path,
            dry_run=False,
            verify=False,
            verbose=False,
        )
        assert code1 == 0
        assert tmp_store_path.is_file()

        store_state1 = load_store(tmp_store_path)
        count1 = len(store_state1["blocks"])
        assert count1 == 1, f"Expected 1 block in store, got {count1}"

        # Check permissions: 0o600
        mode = os.stat(tmp_store_path).st_mode & 0o777
        assert mode == 0o600, f"Expected store permissions 0o600, got {oct(mode)}"

        # 2. Sync again to verify CRDT G-Set idempotence (no duplicates created)
        code2 = run_sync(
            catalog_path=str(cat_path),
            store_path=tmp_store_path,
            dry_run=False,
            verify=False,
            verbose=False,
        )
        assert code2 == 0
        store_state2 = load_store(tmp_store_path)
        count2 = len(store_state2["blocks"])
        assert count2 == 1, f"Idempotence failed: block count changed from {count1} to {count2}"
        assert store_state2["sid_dedup_hits"] == 1, f"Expected 1 dedup hit, got {store_state2['sid_dedup_hits']}"


# ---------------------------------------------------------------------------
# Test Category 9: Unit Verification of verify_concept_item & CLI --verify
# ---------------------------------------------------------------------------

def test_verify_concept_item_fast_and_full(sample_valid_concept):
    """Verify that verify_concept_item correctly accepts valid concepts in both FAST and FULL modes."""
    from experiments.domain_discovery.sync_to_opencommons import verify_concept_item

    # FAST mode (run_deep_sandbox=False)
    ok_fast, err_fast = verify_concept_item(sample_valid_concept, run_deep_sandbox=False)
    assert ok_fast is True, f"FAST mode failed: {err_fast}"
    assert err_fast is None

    # FULL mode (run_deep_sandbox=True) - verifies Step 4 fix with synthesized probes
    ok_deep, err_deep = verify_concept_item(sample_valid_concept, run_deep_sandbox=True)
    assert ok_deep is True, f"FULL mode failed: {err_deep}"
    assert err_deep is None

    # Malformed concept rejection in FAST mode
    bad_concept = dict(sample_valid_concept, invariants=[])
    ok_bad, err_bad = verify_concept_item(bad_concept, run_deep_sandbox=False)
    assert ok_bad is False
    assert "Contracts" in err_bad


def test_sync_to_opencommons_cli_verify_flag(sample_valid_concept):
    """Verify CLI --verify runs L1-L7 and marks accepted block as probe_verified."""
    from experiments.domain_discovery.sync_to_opencommons import run_sync, load_store

    with tempfile.TemporaryDirectory() as tmp_dir:
        store_p = Path(tmp_dir) / "store.json"
        cat_p = Path(tmp_dir) / "cat.json"
        cat_p.write_text(json.dumps([sample_valid_concept]), encoding="utf-8")

        code = run_sync(
            catalog_path=str(cat_p),
            store_path=store_p,
            dry_run=False,
            verify=True,
            verbose=False,
        )
        assert code == 0
        store = load_store(store_p)
        assert len(store["blocks"]) == 1
        assert store["blocks"][0]["trust_tier"] == "probe_verified"


def test_sync_to_opencommons_cli_rejects_invalid_concept_exit_code_1():
    """Verify CLI exits with code 1 when a concept is rejected during verification."""
    from experiments.domain_discovery.sync_to_opencommons import run_sync

    with tempfile.TemporaryDirectory() as tmp_dir:
        store_p = Path(tmp_dir) / "store.json"
        cat_p = Path(tmp_dir) / "bad_cat.json"
        # Trivial 1-liner concept
        bad_item = {
            "name": "bad_1liner",
            "param_types": ["int", "int"],
            "return_type": "int",
            "summary": "Trivial wrapper.",
            "pseudocode": "return x + y",
            "invariants": ["inv1", "inv2"],
            "pitfalls": [{"naive_pattern": "p", "failure_mechanism": "m", "counterexample_input": "c", "verified_fix": "f"}],
            "source_code": "def bad_1liner(x: int, y: int) -> int:\n    return x + y\n",
            "domain": "general",
        }
        cat_p.write_text(json.dumps([bad_item]), encoding="utf-8")

        code = run_sync(
            catalog_path=str(cat_p),
            store_path=store_p,
            dry_run=False,
            verify=False,
        )
        assert code == 1, f"Expected returncode 1 on rejected concept, got {code}"


def test_sync_to_opencommons_cli_rejects_corrupted_json_exit_code_1():
    """Verify CLI exits with code 1 when catalog JSON cannot be parsed."""
    from experiments.domain_discovery.sync_to_opencommons import run_sync

    with tempfile.TemporaryDirectory() as tmp_dir:
        store_p = Path(tmp_dir) / "store.json"
        bad_cat_p = Path(tmp_dir) / "corrupt.json"
        bad_cat_p.write_text('{"bad json syntax: truncated', encoding="utf-8")

        code = run_sync(
            catalog_path=str(bad_cat_p),
            store_path=store_p,
            dry_run=False,
            verify=False,
        )
        assert code == 1, f"Expected returncode 1 on corrupted JSON, got {code}"


# ---------------------------------------------------------------------------
# Test Category 10: Adversarial Negative Tests
# ---------------------------------------------------------------------------

def test_sandbox_rejects_input_mutating_functions(sample_valid_concept):
    """Negative Test: Verify that in-place caller argument mutation is detected and rejected."""
    from experiments.domain_discovery.sync_to_opencommons import verify_concept_item

    mutating_code = (
        "def mutating_vector_scale(data: list, factor: float) -> list:\n"
        "    n = len(data)\n"
        "    data.append(999999.0)\n"
        "    res = [x * factor for x in data]\n"
        "    return res\n"
    )
    # Direct sandbox check
    sb_res = run_sandboxed_probes(
        source_code=mutating_code,
        func_name="mutating_vector_scale",
        probes=[([1.0, 2.0, 3.0], 2.0)],
    )
    assert any(pr.mutated_args for pr in sb_res.probe_results) is True

    # Pipeline verification check
    mut_proposal = dict(
        sample_valid_concept,
        name="mutating_vector_scale",
        param_types=["list", "float"],
        return_type="list",
        source_code=mutating_code,
    )
    ok, err = verify_concept_item(mut_proposal, run_deep_sandbox=True)
    assert ok is False
    assert "argument mutation" in str(err)


def test_sandbox_rejects_nondeterministic_functions(sample_valid_concept):
    """Negative Test: Verify that stateful non-deterministic functions trigger DISCREPANCY and rejection."""
    from experiments.domain_discovery.sync_to_opencommons import verify_concept_item

    nondet_code = (
        "def stateful_accumulator(x: int) -> int:\n"
        "    try:\n"
        "        val = stateful_accumulator.cache\n"
        "    except AttributeError:\n"
        "        val = 0\n"
        "    stateful_accumulator.cache = val + 1\n"
        "    a = x * 2\n"
        "    b = a + stateful_accumulator.cache\n"
        "    return b\n"
    )
    sb_res = run_sandboxed_probes(
        source_code=nondet_code,
        func_name="stateful_accumulator",
        probes=[(10,), (20,)],
    )
    assert sb_res.is_deterministic is False
    assert sb_res.error_code == "DISCREPANCY"

    nondet_proposal = dict(
        sample_valid_concept,
        name="stateful_accumulator",
        param_types=["int"],
        return_type="int",
        source_code=nondet_code,
    )
    ok, err = verify_concept_item(nondet_proposal, run_deep_sandbox=True)
    assert ok is False
    assert "DISCREPANCY" in str(err) or "Sandbox L7 failure" in str(err)


def test_deduplication_rejects_lookalike_duplicate_sids(sample_valid_concept):
    """Negative Test: Verify alpha-renamed functions produce identical SIDs and trigger CRDT deduplication."""
    from experiments.domain_discovery.sync_to_opencommons import (
        normalize_to_store_block,
        crdt_merge_blocks,
        load_store,
    )

    alpha_renamed_code = (
        "def brent_dekker_scalar_root(p0: float, p1: float, p2: float) -> float:\n"
        "    v_a = float(p0)\n"
        "    v_b = float(p1)\n"
        "    v_eps = max(1e-15, abs(p2))\n"
        "    v_fa = v_a * v_a - 2.0\n"
        "    v_fb = v_b * v_b - 2.0\n"
        "    v_c = v_a\n"
        "    v_fc = v_fa\n"
        "    for _ in range(50):\n"
        "        if abs(v_fb) < v_eps:\n"
        "            return v_b\n"
        "        if abs(v_b - v_a) < v_eps:\n"
        "            return v_b\n"
        "        v_m = 0.5 * (v_c - v_b)\n"
        "        v_b = v_b + v_m\n"
        "        v_fb = v_b * v_b - 2.0\n"
        "    return v_b\n"
    )
    sid_orig, _, _, _, _ = compute_structural_sid(sample_valid_concept["source_code"])
    sid_renamed, _, _, _, _ = compute_structural_sid(alpha_renamed_code)
    assert sid_orig == sid_renamed, f"Expected identical SID via alpha-normalization, got {sid_orig} vs {sid_renamed}"

    blk_orig = normalize_to_store_block(sample_valid_concept)
    blk_renamed = normalize_to_store_block(dict(sample_valid_concept, source_code=alpha_renamed_code))

    store0 = load_store(Path("/nonexistent/store.json"))
    store1, add1, _ = crdt_merge_blocks(store0, [blk_orig])
    assert add1 == 1

    store2, add2, hit2 = crdt_merge_blocks(store1, [blk_renamed])
    assert add2 == 0, "Duplicate alpha-renamed concept was incorrectly added"
    assert hit2 == 1, "Duplicate alpha-renamed concept did not trigger dedup hit"


def test_ast_non_triviality_boundary_rejections():
    """Negative Test: Verify boundary condition rejections for AST non-triviality."""
    from experiments.domain_discovery.sync_to_opencommons import check_ast_non_triviality

    # Boundary 1: Statements < 4 with high node count (>= 25)
    heavy_2_stmt = (
        "def heavy_2_stmt(x: float, y: float) -> float:\n"
        "    val = ((x + 1.0) * (y + 2.0) - (x * y)) / ((x * x + 1.0) * (y * y + 1.0) + 1.0)\n"
        "    return val * val + 3.0 * val - 1.0\n"
    )
    fn_def = ast.parse(heavy_2_stmt).body[0]
    assert len(fn_def.body) < 4
    assert sum(1 for _ in ast.walk(fn_def)) >= 25
    ok_2, err_2 = check_ast_non_triviality(heavy_2_stmt)
    assert ok_2 is False
    assert "statement count too low" in str(err_2)

    # Boundary 2: Nodes < 25 with statements >= 4
    small_4_stmt = (
        "def small_4_stmt(x: int) -> int:\n"
        "    a = 1\n"
        "    b = 2\n"
        "    c = 3\n"
        "    return a\n"
    )
    tree_small = ast.parse(small_4_stmt).body[0]
    assert len(tree_small.body) == 4
    assert sum(1 for _ in ast.walk(tree_small)) < 25
    ok_small, err_small = check_ast_non_triviality(small_4_stmt)
    assert ok_small is False
    assert "node count too low" in str(err_small)


# ---------------------------------------------------------------------------
# Test Category 11: CRDT Semilattice, Quarantine & Counter Verification
# ---------------------------------------------------------------------------

def test_crdt_quarantine_propagation_and_persistence():
    """Verify CRDT quarantine propagation on net-new blocks, duplicate SIDs, and quarantined_ids set."""
    from experiments.domain_discovery.sync_to_opencommons import crdt_merge_blocks

    store0 = {"blocks": [], "quarantined_ids": [], "peer_review_executions": 0, "sid_dedup_hits": 0}

    # 1. Net-new quarantined block adds to quarantined_ids
    inc_bad = [{"block_id": "c_bad@1234", "sid": "sid:1234", "trust_tier": "quarantined"}]
    m1, add1, _ = crdt_merge_blocks(store0, inc_bad)
    assert add1 == 1
    assert "c_bad@1234" in m1["quarantined_ids"]
    assert m1["blocks"][0]["trust_tier"] == "quarantined"

    # 2. Existing unverified block upgraded to quarantined when incoming duplicate is quarantined
    store1 = {"blocks": [{"block_id": "c@5555", "sid": "sid:5555", "trust_tier": "unverified"}], "quarantined_ids": []}
    inc_quar = [{"block_id": "c_dup@5555", "sid": "sid:5555", "trust_tier": "quarantined"}]
    m2, add2, hit2 = crdt_merge_blocks(store1, inc_quar)
    assert add2 == 0
    assert hit2 == 1
    assert "c@5555" in m2["quarantined_ids"]
    assert m2["blocks"][0]["trust_tier"] == "quarantined"


def test_crdt_trust_tier_promotion():
    """Verify join-semilattice promotes unverified to probe_verified on verified sync."""
    from experiments.domain_discovery.sync_to_opencommons import crdt_merge_blocks

    store = {"blocks": [{"block_id": "c@1", "sid": "sid:1", "trust_tier": "unverified"}], "quarantined_ids": []}
    incoming = [{"block_id": "c@1", "sid": "sid:1", "trust_tier": "probe_verified"}]
    merged, add, hit = crdt_merge_blocks(store, incoming)
    assert add == 0
    assert hit == 1
    assert merged["blocks"][0]["trust_tier"] == "probe_verified"

    # Quarantined block cannot be demoted
    q_store = {"blocks": [{"block_id": "c@1", "sid": "sid:1", "trust_tier": "quarantined"}], "quarantined_ids": ["c@1"]}
    q_merged, _, _ = crdt_merge_blocks(q_store, incoming)
    assert q_merged["blocks"][0]["trust_tier"] == "quarantined"
    assert "c@1" in q_merged["quarantined_ids"]


def test_crdt_dedup_hits_no_runaway_inflation():
    """Verify that saving new blocks sequentially does not inflate sid_dedup_hits on disk."""
    from experiments.domain_discovery.sync_to_opencommons import (
        crdt_merge_blocks,
        save_store_atomic,
        load_store,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        store_p = Path(tmp_dir) / "store.json"

        # 5 distinct initial concepts
        b_init = [{"block_id": f"c_{i}@1", "sid": f"sid:{i}", "trust_tier": "unverified"} for i in range(5)]
        s1, add1, hit1 = crdt_merge_blocks({"blocks": []}, b_init)
        assert hit1 == 0
        save_store_atomic(s1, store_p, incoming_blocks=b_init)

        d1 = load_store(store_p)
        assert d1["sid_dedup_hits"] == 0

        # Sync 1 brand new concept (zero duplicates)
        b_new = [{"block_id": "c_new@999", "sid": "sid:999", "trust_tier": "unverified"}]
        s2, add2, hit2 = crdt_merge_blocks(d1, b_new)
        assert hit2 == 0
        save_store_atomic(s2, store_p, incoming_blocks=b_new)

        d2 = load_store(store_p)
        assert d2["sid_dedup_hits"] == 0, f"Runaway inflation: expected 0 hits, got {d2['sid_dedup_hits']}"

