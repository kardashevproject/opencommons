"""Streaming High-Dimensional Telemetry Anomaly Detection & Non-Parametric Hypothesis Testing Engine.

Composes 6 verified OpenCommons concepts from the statistics_and_information domain:
1. `welford_skewness_kurtosis`: Online stream updates for 1st-4th central moments.
2. `t_digest_centroid_merge`: Bounded streaming quantile centroids sketch.
3. `median_absolute_deviation_mad`: Outlier-resistant robust scale estimation.
4. `kullback_leibler_discrete_div`: Directional relative entropy information divergence.
5. `jensen_shannon_metric_distance`: Metric distance between empirical telemetry distributions.
6. `benjamini_hochberg_fdr_step`: Multiple hypothesis testing False Discovery Rate filter.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable


def _load_domain_functions() -> dict[str, Callable[..., Any]]:
    catalog_path = Path(__file__).resolve().parent.parent.parent.parent / "data" / "domains" / "statistics_and_information.json"
    with open(catalog_path, "r", encoding="utf-8") as f:
        concepts = json.load(f)
    env: dict[str, Any] = {"math": math}
    for c in concepts:
        code_obj = compile(c["source_code"], f"<{c['name']}>", "exec")
        exec(code_obj, env)
    return {c["name"]: env[c["name"]] for c in concepts}


_FUNCS = _load_domain_functions()
welford_skewness_kurtosis = _FUNCS["welford_skewness_kurtosis"]
t_digest_centroid_merge = _FUNCS["t_digest_centroid_merge"]
median_absolute_deviation_mad = _FUNCS["median_absolute_deviation_mad"]
kullback_leibler_discrete_div = _FUNCS["kullback_leibler_discrete_div"]
jensen_shannon_metric_distance = _FUNCS["jensen_shannon_metric_distance"]
benjamini_hochberg_fdr_step = _FUNCS["benjamini_hochberg_fdr_step"]


class StreamingTelemetryAnomalyDetector:
    """Multi-channel streaming telemetry anomaly detection engine."""

    def __init__(
        self,
        n_channels: int = 8,
        window_size: int = 50,
        fdr_q: float = 0.05,
    ) -> None:
        self.n_channels = n_channels
        self.window_size = window_size
        self.fdr_q = fdr_q

        # Channel states: count, mean, m2, m3, m4
        self.moments = [(0, 0.0, 0.0, 0.0, 0.0) for _ in range(n_channels)]
        self.windows: list[list[float]] = [[] for _ in range(n_channels)]
        self.baselines: list[list[float]] = [[] for _ in range(n_channels)]

    def record_baseline(self, channel: int, samples: list[float]) -> None:
        """Stores baseline reference samples for a channel."""
        self.baselines[channel] = list(samples)

    def ingest_sample(self, channel: int, value: float) -> None:
        """Ingests a new telemetry sample for a given channel."""
        cnt, m, m2, m3, m4 = self.moments[channel]
        self.moments[channel] = welford_skewness_kurtosis(cnt, m, m2, m3, m4, value)
        self.windows[channel].append(value)
        if len(self.windows[channel]) > self.window_size:
            self.windows[channel].pop(0)

    def _discretize_histogram(
        self,
        samples_a: list[float],
        samples_b: list[float],
        n_bins: int = 10,
    ) -> tuple[list[float], list[float]]:
        all_vals = samples_a + samples_b
        if not all_vals:
            return ([1.0], [1.0])
        min_v = min(all_vals)
        max_v = max(all_vals)
        if abs(max_v - min_v) < 1e-9:
            return ([1.0], [1.0])

        step = (max_v - min_v) / float(n_bins)
        counts_a = [0.0] * n_bins
        counts_b = [0.0] * n_bins

        for v in samples_a:
            b_idx = min(n_bins - 1, max(0, int((v - min_v) / step)))
            counts_a[b_idx] += 1.0

        for v in samples_b:
            b_idx = min(n_bins - 1, max(0, int((v - min_v) / step)))
            counts_b[b_idx] += 1.0

        # Laplace smoothing
        sa = sum(counts_a) + float(n_bins)
        sb = sum(counts_b) + float(n_bins)
        pa = [(c + 1.0) / sa for c in counts_a]
        pb = [(c + 1.0) / sb for c in counts_b]
        return (pa, pb)

    def evaluate_anomalies(self) -> dict[str, Any]:
        """Runs full multi-channel distribution shift detection and FDR filtering."""
        p_values: list[float] = []
        js_distances: list[float] = []
        kl_divs: list[float] = []
        mads: list[float] = []
        centroid_summaries: list[dict[str, Any]] = []

        for ch in range(self.n_channels):
            cur_win = self.windows[ch]
            base = self.baselines[ch]

            # 1. Robust scale
            mad_val = median_absolute_deviation_mad(cur_win, scale_normal=True) if cur_win else 0.0
            mads.append(mad_val)

            # 2. Centroid merge quantile compression
            if cur_win:
                means = list(cur_win)
                weights = [1.0] * len(cur_win)
                c_means, c_weights = t_digest_centroid_merge(means, weights, max_centroids=8)
                centroid_summaries.append({"means": c_means, "weights": c_weights})
            else:
                centroid_summaries.append({"means": [], "weights": []})

            # 3. Distributional Divergence (KL & JS)
            if base and cur_win:
                pa, pb = self._discretize_histogram(base, cur_win, n_bins=10)
                js = jensen_shannon_metric_distance(pa, pb)
                kl = kullback_leibler_discrete_div(pb, pa)
            else:
                js = 0.0
                kl = 0.0

            js_distances.append(js)
            kl_divs.append(kl)

            # Estimate p-value under null distribution (asymptotically based on JS distance)
            # Under null, JS distance is small (< 0.15). We transform JS distance into an upper tail p-value.
            z = max(0.0, (js - 0.10) / 0.08)
            # Normal upper tail p-value approximation via complementary error function
            p_val = 0.5 * math.erfc(z / math.sqrt(2.0))
            p_values.append(max(1e-9, min(1.0, p_val)))

        # 4. Benjamini-Hochberg FDR control
        fdr_rejections = benjamini_hochberg_fdr_step(p_values, self.fdr_q)

        return {
            "n_channels": self.n_channels,
            "moments": self.moments,
            "mads": mads,
            "js_distances": js_distances,
            "kl_divergences": kl_divs,
            "p_values": p_values,
            "fdr_rejections": fdr_rejections,
            "anomalous_channels": [ch for ch, rej in enumerate(fdr_rejections) if rej],
        }


def run_streaming_telemetry_benchmark() -> dict[str, Any]:
    """Generates synthetic multi-channel streaming telemetry and executes detection."""
    detector = StreamingTelemetryAnomalyDetector(n_channels=6, window_size=50, fdr_q=0.05)

    # Establish baseline for all 6 channels: normal N(10.0, 1.0)
    import random
    rng = random.Random(42)

    for ch in range(6):
        base_samples = [rng.gauss(10.0, 1.0) for _ in range(100)]
        detector.record_baseline(ch, base_samples)

    # Ingest streaming data:
    # Channels 0-3: Normal operation (same distribution N(10.0, 1.0))
    # Channels 4-5: Injected anomaly (mean shift to 15.0 and variance explosion to 4.0)
    for _ in range(60):
        for ch in range(4):
            detector.ingest_sample(ch, rng.gauss(10.0, 1.0))
        for ch in range(4, 6):
            detector.ingest_sample(ch, rng.gauss(15.0, 2.5))

    return detector.evaluate_anomalies()
