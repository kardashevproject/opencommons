"""Statistical testing, bootstrap confidence intervals, and automated confound auditing."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Sequence
import numpy as np


def _pearsonr(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Compute Pearson correlation coefficient and two-tailed p-value using pure NumPy/math."""
    n = len(x)
    if n < 3:
        return 0.0, 1.0
    xm = x - np.mean(x)
    ym = y - np.mean(y)
    denom = float(np.linalg.norm(xm) * np.linalg.norm(ym))
    if denom < 1e-15:
        return 0.0, 1.0
    r = float(np.clip(np.dot(xm, ym) / denom, -1.0, 1.0))
    if abs(r) >= 1.0 - 1e-15:
        return r, 0.0
    df = n - 2
    t_stat = abs(r) * math.sqrt(df / max(1.0 - r * r, 1e-15))
    # Normal/Cornish-Fisher approximation for two-tailed Student-t p-value (accurate for df >= 2)
    z = t_stat * (1.0 - 1.0 / (4.0 * df)) / math.sqrt(1.0 + (t_stat * t_stat) / (2.0 * df))
    p_val = float(math.erfc(z / math.sqrt(2.0)))
    return r, p_val


@dataclass(frozen=True)
class CIResult:
    """Point estimate with bootstrap confidence interval."""

    estimate: float
    ci_low: float
    ci_high: float
    std_err: float
    n: int
    confidence: float = 0.95

    def format_latex(self, precision: int = 2) -> str:
        """Format as point estimate with 95% CI for LaTeX tables."""
        fmt = f"{{:.{precision}f}}"
        return f"{fmt.format(self.estimate)} [{fmt.format(self.ci_low)}, {fmt.format(self.ci_high)}]"


@dataclass(frozen=True)
class ComparisonResult:
    """Two-arm statistical comparison (paired or unpaired)."""

    mean_a: float
    mean_b: float
    diff: float
    ratio: float
    ci_diff_low: float
    ci_diff_high: float
    p_value: float
    cohens_d: float
    n_a: int
    n_b: int
    paired: bool


@dataclass(frozen=True)
class ConfoundAuditResult:
    """Audit checking whether a target signal is confounded by a nuisance covariate."""

    covariate_name: str
    raw_correlation: float
    raw_p_value: float
    partial_correlation: float
    partial_p_value: float
    signal_retained_fraction: float
    is_confounded: bool


def bootstrap_ci(
    values: Sequence[float],
    stat_fn: Callable[[np.ndarray], float] = np.mean,
    confidence: float = 0.95,
    n_boot: int = 5000,
    seed: int = 42,
) -> CIResult:
    """Compute non-parametric bootstrap confidence interval."""
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    if n == 0:
        raise ValueError("Cannot compute bootstrap CI on empty sequence.")
    estimate = float(stat_fn(arr))
    if n == 1:
        return CIResult(estimate, estimate, estimate, 0.0, 1, confidence)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_stats = np.array([stat_fn(arr[i]) for i in idx], dtype=np.float64)
    alpha = (1.0 - confidence) / 2.0
    ci_low = float(np.quantile(boot_stats, alpha))
    ci_high = float(np.quantile(boot_stats, 1.0 - alpha))
    std_err = float(np.std(boot_stats, ddof=1))
    return CIResult(
        estimate=estimate,
        ci_low=ci_low,
        ci_high=ci_high,
        std_err=std_err,
        n=n,
        confidence=confidence,
    )


def compare_arms(
    arm_a: Sequence[float],
    arm_b: Sequence[float],
    paired: bool = True,
    n_boot: int = 5000,
    n_perm: int = 10000,
    seed: int = 42,
) -> ComparisonResult:
    """Compare two experimental arms with bootstrap diff CI and exact permutation p-value."""
    a = np.asarray(arm_a, dtype=np.float64)
    b = np.asarray(arm_b, dtype=np.float64)
    if len(a) == 0 or len(b) == 0:
        raise ValueError("Arms must be non-empty.")
    if paired and len(a) != len(b):
        raise ValueError(f"Paired comparison requires equal lengths, got {len(a)} vs {len(b)}.")

    mean_a = float(np.mean(a))
    mean_b = float(np.mean(b))
    diff = mean_a - mean_b
    ratio = mean_a / mean_b if abs(mean_b) > 1e-12 else float("inf")

    rng = np.random.default_rng(seed)

    if paired:
        deltas = a - b
        n = len(deltas)
        ci = bootstrap_ci(deltas, np.mean, n_boot=n_boot, seed=seed)
        std_d = float(np.std(deltas, ddof=1)) if n > 1 else 0.0
        cohens_d = diff / std_d if std_d > 1e-12 else 0.0

        # Sign-flip permutation test
        signs = rng.choice([-1.0, 1.0], size=(n_perm, n))
        perm_means = np.mean(signs * deltas, axis=1)
        p_val = float((np.sum(np.abs(perm_means) >= abs(diff)) + 1) / (n_perm + 1))
        return ComparisonResult(
            mean_a=mean_a,
            mean_b=mean_b,
            diff=diff,
            ratio=ratio,
            ci_diff_low=ci.ci_low,
            ci_diff_high=ci.ci_high,
            p_value=p_val,
            cohens_d=cohens_d,
            n_a=len(a),
            n_b=len(b),
            paired=True,
        )
    else:
        n_a, n_b = len(a), len(b)
        idx_a = rng.integers(0, n_a, size=(n_boot, n_a))
        idx_b = rng.integers(0, n_b, size=(n_boot, n_b))
        boot_diffs = np.mean(a[idx_a], axis=1) - np.mean(b[idx_b], axis=1)
        ci_low = float(np.quantile(boot_diffs, 0.025))
        ci_high = float(np.quantile(boot_diffs, 0.975))

        pooled_var = (
            ((n_a - 1) * np.var(a, ddof=1) + (n_b - 1) * np.var(b, ddof=1)) / max(n_a + n_b - 2, 1)
            if (n_a + n_b > 2)
            else 0.0
        )
        pooled_sd = float(np.sqrt(pooled_var))
        cohens_d = diff / pooled_sd if pooled_sd > 1e-12 else 0.0

        combined = np.concatenate([a, b])
        perm_diffs = np.empty(n_perm, dtype=np.float64)
        for i in range(n_perm):
            perm = rng.permutation(combined)
            perm_diffs[i] = np.mean(perm[:n_a]) - np.mean(perm[n_a:])
        p_val = float((np.sum(np.abs(perm_diffs) >= abs(diff)) + 1) / (n_perm + 1))
        return ComparisonResult(
            mean_a=mean_a,
            mean_b=mean_b,
            diff=diff,
            ratio=ratio,
            ci_diff_low=ci_low,
            ci_diff_high=ci_high,
            p_value=p_val,
            cohens_d=cohens_d,
            n_a=n_a,
            n_b=n_b,
            paired=False,
        )


def holm_bonferroni(p_values: Sequence[float]) -> list[float]:
    """Adjust p-values for multiple hypothesis testing using the Holm-Bonferroni step-down method."""
    m = len(p_values)
    if m == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adj = min(1.0, p * (m - rank))
        running_max = max(running_max, adj)
        adjusted[orig_idx] = running_max
    return adjusted


def audit_confound(
    predictor: Sequence[float],
    outcome: Sequence[float],
    covariate: Sequence[float],
    covariate_name: str = "length",
    min_retained_fraction: float = 0.50,
) -> ConfoundAuditResult:
    """Audit whether the correlation between predictor and outcome is confounded by a nuisance covariate.

    Directly prevents errors like F-021/TF-008 (where 94% of a 'correctness' probe was answer length).
    """
    x = np.asarray(predictor, dtype=np.float64)
    y = np.asarray(outcome, dtype=np.float64)
    z = np.asarray(covariate, dtype=np.float64)
    if not (len(x) == len(y) == len(z)) or len(x) < 4:
        raise ValueError("predictor, outcome, and covariate must have equal length >= 4.")

    raw_r, raw_p = _pearsonr(x, y)

    # Residualize both x and y on z via ordinary least squares
    z_design = np.column_stack([np.ones_like(z), z])
    beta_x, _, _, _ = np.linalg.lstsq(z_design, x, rcond=None)
    beta_y, _, _, _ = np.linalg.lstsq(z_design, y, rcond=None)
    res_x = x - z_design @ beta_x
    res_y = y - z_design @ beta_y

    if np.std(res_x) < 1e-12 or np.std(res_y) < 1e-12:
        partial_r, partial_p = 0.0, 1.0
    else:
        partial_r, partial_p = _pearsonr(res_x, res_y)

    retained = abs(float(partial_r)) / max(abs(float(raw_r)), 1e-12)
    is_confounded = (retained < min_retained_fraction) or (float(partial_p) > 0.05 and float(raw_p) <= 0.05)

    return ConfoundAuditResult(
        covariate_name=covariate_name,
        raw_correlation=float(raw_r),
        raw_p_value=float(raw_p),
        partial_correlation=float(partial_r),
        partial_p_value=float(partial_p),
        signal_retained_fraction=float(retained),
        is_confounded=bool(is_confounded),
    )
