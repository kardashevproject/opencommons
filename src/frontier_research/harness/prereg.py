"""Pre-registration schema and hypothesis verification protocol."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

ComparisonOp = Literal[">=", "<=", ">", "<", "=="]
VerdictType = Literal["SUPPORTED", "REFUTED", "CONFOUNDED"]


@dataclass(frozen=True)
class HypothesisSpec:
    """A pre-registered quantitative hypothesis with a falsifiable threshold."""

    id: str
    statement: str
    metric_name: str
    op: ComparisonOp
    threshold: float
    max_p_value: float = 0.05
    required_confound_checks: tuple[str, ...] = ()

    def evaluate(
        self,
        observed_value: float,
        p_value: float = 1.0,
        confound_flags: dict[str, bool] | None = None,
    ) -> VerdictType:
        """Evaluate whether the observed metric passes the pre-registered bar."""
        if confound_flags:
            for c_name in self.required_confound_checks:
                if confound_flags.get(c_name, False):
                    return "CONFOUNDED"

        if p_value > self.max_p_value:
            return "REFUTED"

        if self.op == ">=":
            passed = observed_value >= self.threshold
        elif self.op == "<=":
            passed = observed_value <= self.threshold
        elif self.op == ">":
            passed = observed_value > self.threshold
        elif self.op == "<":
            passed = observed_value < self.threshold
        elif self.op == "==":
            passed = abs(observed_value - self.threshold) < 1e-9
        else:
            raise ValueError(f"Unsupported operator: {self.op}")

        return "SUPPORTED" if passed else "REFUTED"


@dataclass(frozen=True)
class ExperimentSpec:
    """Full pre-registration specification for an experiment."""

    exp_id: str
    title: str
    track: Literal["concept_commons", "llm_mechanistic", "joint"]
    description: str
    arms: tuple[str, ...]
    hypotheses: tuple[HypothesisSpec, ...]
    seeds: tuple[int, ...] = (42, 43, 44)
    confound_variables: tuple[str, ...] = ("length", "latency_sec")
    metadata: dict[str, str] = field(default_factory=dict)

    def validate(self) -> None:
        """Ensure pre-registration spec is well-formed before execution."""
        if not self.exp_id.startswith("R-"):
            raise ValueError(f"Experiment ID must start with 'R-' (got '{self.exp_id}').")
        if len(self.arms) < 2:
            raise ValueError(f"Experiment '{self.exp_id}' must specify at least 2 arms (control + treatment).")
        if not self.hypotheses:
            raise ValueError(f"Experiment '{self.exp_id}' must pre-register at least 1 falsifiable hypothesis.")
        seen_h: set[str] = set()
        for h in self.hypotheses:
            if h.id in seen_h:
                raise ValueError(f"Duplicate hypothesis ID '{h.id}' in '{self.exp_id}'.")
            seen_h.add(h.id)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
