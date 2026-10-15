"""Benchmark evaluation and statistical verification utilities."""

from frontier_research.harness.prereg import ExperimentSpec, HypothesisSpec, VerdictType
from frontier_research.harness.runner import ExperimentRunRecord, HypothesisOutcome, ResearchHarness
from frontier_research.harness.stats import (
    CIResult,
    ComparisonResult,
    ConfoundAuditResult,
    audit_confound,
    bootstrap_ci,
    compare_arms,
    holm_bonferroni,
)

__all__ = [
    "CIResult",
    "ComparisonResult",
    "ConfoundAuditResult",
    "ExperimentRunRecord",
    "ExperimentSpec",
    "HypothesisOutcome",
    "HypothesisSpec",
    "ResearchHarness",
    "VerdictType",
    "audit_confound",
    "bootstrap_ci",
    "compare_arms",
    "holm_bonferroni",
]
