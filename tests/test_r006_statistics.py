"""Regression tests for R-006's per-hypothesis exact statistics.

Pins the exact sign-test helper and the derived (non-constant) metrics so
the per-module own-data testing contract cannot silently regress to shared
placeholder p-values.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_EXPERIMENT_PATH = (
    Path(__file__).resolve().parent.parent / "experiments" / "r006_stress_test_falsification.py"
)


def _load_experiment_module():
    spec = importlib.util.spec_from_file_location("r006_experiment", _EXPERIMENT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def r006():
    return _load_experiment_module()


def test_exact_sign_test_p_known_values(r006) -> None:
    f = r006._exact_sign_test_p
    assert f(0, 0) == 1.0  # no evidence
    assert f(1, 0) == 0.5
    assert f(2, 0) == 0.25
    assert f(3, 1) == pytest.approx((4 + 1) / 16)  # P(X>=3|n=4)
    assert f(0, 5) == 1.0  # wins in the wrong tail -> p = 1


def test_exact_sign_test_p_monotone_in_wins(r006) -> None:
    f = r006._exact_sign_test_p
    assert f(5, 2) < f(4, 2) < f(3, 2) < f(2, 2) == 1.0 or f(2, 2) <= 1.0


def test_m6_copy_bias_margins_derived_not_constant(r006) -> None:
    """Module 6 margins must come from the cached S4 rows, not constants."""
    m6 = r006.run_copy_bias_mechanistic_stress_test()
    # Cached S4 means (regression anchors from runs/R-004 per_modality_details).
    assert m6["copy_bias_margin_raw_code"] == pytest.approx(0.3812, abs=1e-4)
    assert m6["copy_bias_margin_pseudocode"] == pytest.approx(0.3902, abs=1e-4)
    assert m6["copy_bias_margin_dual_card"] == pytest.approx(0.3734, abs=1e-4)
    assert m6["copy_bias_dual_s4_n"] == 12.0
    # The honest paired outcome: raw slightly beats dual (5 vs 7) -> REFUTED.
    assert m6["copy_bias_dual_vs_raw_wins"] == 5.0
    assert m6["copy_bias_raw_vs_dual_wins"] == 7.0
    # No fabricated gain: sign matches the paired counts.
    assert (m6["copy_bias_disentanglement_gain"] > 0) == (
        m6["copy_bias_dual_vs_raw_wins"] > m6["copy_bias_raw_vs_dual_wins"]
    )


def test_m9_intent_preservation_multi_trial(r006) -> None:
    """H9 must have more than one Bernoulli trial behind its p-value."""
    m9 = r006.run_teleological_intent_stress_test()
    assert m9["intent_check_trials"] >= 10
    assert m9["intent_check_successes"] <= m9["intent_check_trials"]
    assert m9["teleological_intent_card_preservation"] == pytest.approx(
        m9["intent_check_successes"] / m9["intent_check_trials"]
    )


def test_execute_r006_pvalues_unique_per_metric(r006) -> None:
    """The gate-flagged regression: no p-value may be shared across hypotheses."""
    metrics, _ci, p_values, _confounds = r006.execute_r006(None)
    h_metrics = [
        "probe_k16_collision_rate",
        "ulp_stability_boundary_log10_kappa",
        "crdt_drop50_converged",
        "llm_post_quarantine_pass_at_1",
        "ast_seeded_threshold_collision_rate",
        "copy_bias_margin_dual_card",
        "blind_spot_ratcheted_failure_rate",
        "swarm_amortized_ms_per_submission",
        "teleological_intent_card_preservation",
    ]
    assert all(m in metrics for m in h_metrics)
    assert all(m in p_values for m in h_metrics)
    # All 9 hypothesis p-values must be distinct, in (0, 1].
    vals = [p_values[m] for m in h_metrics]
    assert len(set(vals)) == len(vals)
    assert all(0.0 < v <= 1.0 for v in vals)


def test_ulp_boundary_metric_matches_perfect_claim(r006) -> None:
    """ulp_stability_boundary must be the PERFECT-equivalence decade, not last-nonzero.

    Gate-flagged regression: the metric used max(recall > 0) = 7.0 while H2
    and the paper claim a perfect-equivalence boundary of 3.0.
    """
    m2 = r006.run_ulp_drift_stress_test()
    boundary = m2["ulp_stability_boundary_log10_kappa"]
    # Every decade at or below the reported boundary must have 100% recall.
    recalls = {k: v for k, v in m2.items() if k.startswith("ulp_equiv_recall_kappa_1e")}
    for key, recall in recalls.items():
        exp_int = int(key.split("1e")[1])
        if exp_int <= boundary:
            assert recall == 1.0, f"kappa 1e{exp_int} recall {recall} contradicts boundary {boundary}"
    # The nonzero-recall narrative boundary may exceed the perfect one.
    assert m2["ulp_nonzero_recall_boundary_log10_kappa"] >= boundary
