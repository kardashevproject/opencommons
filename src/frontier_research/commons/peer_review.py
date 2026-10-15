"""Asynchronous 'Open-Science Peer Review' refutation and quarantine engine."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable, Literal

from frontier_research.commons.fingerprint import (
    FingerprintResult,
    _canonical_serialize,
    generate_canonical_probes,
)

TrustTier = Literal["probe_verified", "peer_reviewed", "quarantined"]


@dataclass(frozen=True)
class PeerReviewVerdict:
    """Outcome of the asynchronous background peer-review audit on a contributed concept block."""

    trust_tier: TrustTier
    passed: bool
    reasons: tuple[str, ...]
    counterexample: str | None = None


def run_async_peer_review(
    source_code: str,
    fp_result: FingerprintResult,
    contract_io: list[tuple[tuple[Any, ...], Any]] | None = None,
    invariant_fn: Callable[[tuple[Any, ...], Any], bool] | None = None,
    reference_fn: Callable[..., Any] | None = None,
    allow_mutation: bool = False,
    extra_globals: dict[str, Any] | None = None,
) -> PeerReviewVerdict:
    """Audit a contributed block off the critical path for subtle bugs, spec gaming, and edge-case failures.

    Implements the lessons from F12/F19 (spec gaming), F32/F33/F65 (boundary refutation),
    F78 (checksum/constant bugs), and F80 (unintended in-place argument mutation).
    """
    reasons: list[str] = []

    # Gate 1: Zero valid executions on canonical probe suite
    if fp_result.valid_probe_count == 0 and not contract_io:
        return PeerReviewVerdict(
            trust_tier="quarantined",
            passed=False,
            reasons=("zero_valid_probes: function raised exceptions on all 16 canonical probes",),
        )

    # Gate 2: Degenerate constant output (spec-gaming / stub implementation)
    if fp_result.is_constant_output:
        return PeerReviewVerdict(
            trust_tier="quarantined",
            passed=False,
            reasons=(f"degenerate_constant_output: returned {fp_result.sample_outputs[0]} on all probes",),
        )

    # Gate 3: Silent in-place mutation of caller arguments (F80)
    if fp_result.mutates_arguments and not allow_mutation and fp_result.return_type.lower() != "none":
        return PeerReviewVerdict(
            trust_tier="quarantined",
            passed=False,
            reasons=("silent_argument_mutation: mutated caller input container in-place while returning a value",),
        )

    ns: dict[str, Any] = {}
    if extra_globals:
        ns.update(extra_globals)
    exec(compile(source_code, "<peer_review>", "exec"), ns)  # noqa: S102
    fn: Callable[..., Any] = ns[fp_result.func_name]

    # Gate 4: Explicit contract I/O assertions
    if contract_io:
        for args, expected in contract_io:
            try:
                actual = fn(*copy.deepcopy(args))
                if _canonical_serialize(actual) != _canonical_serialize(expected):
                    return PeerReviewVerdict(
                        trust_tier="quarantined",
                        passed=False,
                        reasons=("contract_io_mismatch",),
                        counterexample=f"args={args!r} -> got {actual!r}, expected {expected!r}",
                    )
            except Exception as exc:  # noqa: BLE001
                return PeerReviewVerdict(
                    trust_tier="quarantined",
                    passed=False,
                    reasons=(f"contract_io_exception:{type(exc).__name__}",),
                    counterexample=f"args={args!r} raised {exc!r}",
                )

    # Gate 5: Extended 32-probe adversarial/boundary fuzzing (invariant or differential check)
    extended_probes = generate_canonical_probes(fp_result.param_types, n_probes=32)
    for probe_args in extended_probes:
        if invariant_fn is not None:
            try:
                out = fn(*copy.deepcopy(probe_args))
                if not invariant_fn(probe_args, out):
                    return PeerReviewVerdict(
                        trust_tier="quarantined",
                        passed=False,
                        reasons=("invariant_violation",),
                        counterexample=f"args={probe_args!r} -> output={out!r} violated invariant",
                    )
            except Exception as exc:  # noqa: BLE001
                return PeerReviewVerdict(
                    trust_tier="quarantined",
                    passed=False,
                    reasons=(f"adversarial_probe_crash:{type(exc).__name__}",),
                    counterexample=f"args={probe_args!r} raised {exc!r}",
                )

        if reference_fn is not None:
            try:
                ref_out = reference_fn(*copy.deepcopy(probe_args))
                ref_err = None
            except Exception as exc:  # noqa: BLE001
                ref_out = None
                ref_err = type(exc).__name__

            try:
                cand_out = fn(*copy.deepcopy(probe_args))
                cand_err = None
            except Exception as exc:  # noqa: BLE001
                cand_out = None
                cand_err = type(exc).__name__

            if ref_err != cand_err or (
                ref_err is None and _canonical_serialize(cand_out) != _canonical_serialize(ref_out)
            ):
                return PeerReviewVerdict(
                    trust_tier="quarantined",
                    passed=False,
                    reasons=("differential_refutation_mismatch",),
                    counterexample=(
                        f"args={probe_args!r} -> candidate={cand_out!r} (err={cand_err}), "
                        f"reference={ref_out!r} (err={ref_err})"
                    ),
                )

    reasons.append("passed_all_peer_review_gates")
    return PeerReviewVerdict(
        trust_tier="peer_reviewed",
        passed=True,
        reasons=tuple(reasons),
        counterexample=None,
    )
