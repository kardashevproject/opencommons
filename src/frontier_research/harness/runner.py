"""Benchmark evaluation runner and LaTeX table generator."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Callable

from frontier_research.harness.prereg import ExperimentSpec, VerdictType
from frontier_research.harness.stats import CIResult, ConfoundAuditResult


@dataclass(frozen=True)
class HypothesisOutcome:
    """Evaluated outcome for a single pre-registered hypothesis."""

    hypothesis_id: str
    statement: str
    metric_name: str
    observed_value: float
    op: str
    threshold: float
    p_value: float
    verdict: VerdictType


@dataclass(frozen=True)
class ExperimentRunRecord:
    """Complete immutable provenance and result record for an experiment run."""

    exp_id: str
    title: str
    track: str
    started_at: str
    completed_at: str
    elapsed_sec: float
    metrics: dict[str, float]
    ci_metrics: dict[str, dict[str, float]]
    p_values: dict[str, float]
    confound_audits: dict[str, dict[str, object]]
    outcomes: list[HypothesisOutcome]
    artifacts_sha256: str


class ResearchHarness:
    """Orchestrates benchmark suites, audits confounds, and emits LaTeX paper tables."""

    def __init__(self, workspace_root: Path | str | None = None) -> None:
        self.root = Path(workspace_root) if workspace_root is not None else Path(__file__).resolve().parents[3]
        self.runs_dir = self.root / "runs"
        self.paper_tables_dir = self.root / "paper" / "tables"

    def run_experiment(
        self,
        spec: ExperimentSpec,
        execute_fn: Callable[
            [ExperimentSpec],
            tuple[
                dict[str, float],
                dict[str, CIResult],
                dict[str, float],
                dict[str, ConfoundAuditResult],
            ],
        ],
    ) -> ExperimentRunRecord:
        """Validate specification, execute benchmark, evaluate hypotheses, and write artifacts."""
        spec.validate()
        started_iso = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        t0 = time.perf_counter()

        metrics, ci_results, p_values, confound_audits = execute_fn(spec)

        elapsed = round(time.perf_counter() - t0, 4)
        completed_iso = time.strftime("%Y-%m-%dT%H:%M:%S%z")

        confound_flags = {name: audit.is_confounded for name, audit in confound_audits.items()}

        outcomes: list[HypothesisOutcome] = []
        for h in spec.hypotheses:
            if h.metric_name not in metrics:
                raise KeyError(
                    f"Pre-registered hypothesis '{h.id}' requires metric '{h.metric_name}', "
                    f"which was not returned by execute_fn."
                )
            obs = float(metrics[h.metric_name])
            pval = float(p_values.get(h.metric_name, 0.0))
            verdict = h.evaluate(obs, p_value=pval, confound_flags=confound_flags)
            outcomes.append(
                HypothesisOutcome(
                    hypothesis_id=h.id,
                    statement=h.statement,
                    metric_name=h.metric_name,
                    observed_value=obs,
                    op=h.op,
                    threshold=h.threshold,
                    p_value=pval,
                    verdict=verdict,
                )
            )

        ci_dict = {k: asdict(v) for k, v in ci_results.items()}
        confound_dict = {k: asdict(v) for k, v in confound_audits.items()}

        payload_bytes = json.dumps(
            {
                "exp_id": spec.exp_id,
                "metrics": metrics,
                "ci_metrics": ci_dict,
                "p_values": p_values,
                "outcomes": [asdict(o) for o in outcomes],
            },
            sort_keys=True,
        ).encode("utf-8")
        digest = hashlib.sha256(payload_bytes).hexdigest()

        record = ExperimentRunRecord(
            exp_id=spec.exp_id,
            title=spec.title,
            track=spec.track,
            started_at=started_iso,
            completed_at=completed_iso,
            elapsed_sec=elapsed,
            metrics=metrics,
            ci_metrics=ci_dict,
            p_values=p_values,
            confound_audits=confound_dict,
            outcomes=outcomes,
            artifacts_sha256=digest,
        )

        self._persist_run(spec, record, ci_results)
        return record

    def _persist_run(
        self,
        spec: ExperimentSpec,
        record: ExperimentRunRecord,
        ci_results: dict[str, CIResult],
    ) -> None:
        exp_dir = self.runs_dir / spec.exp_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        self.paper_tables_dir.mkdir(parents=True, exist_ok=True)

        results_path = exp_dir / "results.json"
        results_path.write_text(json.dumps(asdict(record), indent=2, sort_keys=True) + "\n", encoding="utf-8")

        tex_lines = [
            f"% Auto-generated from {record.exp_id} (SHA-256: {record.artifacts_sha256[:12]})",
            "\\begin{tabular}{llrrr}",
            "\\toprule",
            "Hypothesis & Metric & Observed (95\\% CI) & Threshold & Verdict \\\\",
            "\\midrule",
        ]
        for o in record.outcomes:
            m_escaped = o.metric_name.replace("_", "\\_")
            if o.metric_name in ci_results:
                obs_str = ci_results[o.metric_name].format_latex(precision=3)
            else:
                obs_str = f"{o.observed_value:.3f}"
            op_tex = {">=": "$\\ge$", "<=": "$\\le$", ">": "$>$", "<": "$<$", "==": "$=$"}.get(o.op, o.op)
            tex_lines.append(
                f"{o.hypothesis_id} & \\texttt{{{m_escaped}}} & {obs_str} & {op_tex} {o.threshold:.3f} & \\textbf{{{o.verdict}}} \\\\"
            )
        tex_lines.extend(["\\bottomrule", "\\end{tabular}"])
        (self.paper_tables_dir / f"{spec.exp_id}.tex").write_text("\n".join(tex_lines) + "\n", encoding="utf-8")
