"""Pytest test suite for Statistics & Information Theory Streaming Telemetry Benchmark."""

import json
from pathlib import Path
import sys
import pytest

repo_root = str(Path(__file__).resolve().parent.parent.parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from experiments.domain_discovery.statistics_and_information.benchmark import (
    StreamingTelemetryAnomalyDetector,
    run_streaming_telemetry_benchmark,
    welford_skewness_kurtosis,
    jensen_shannon_metric_distance,
    kullback_leibler_discrete_div,
    benjamini_hochberg_fdr_step,
    median_absolute_deviation_mad,
    t_digest_centroid_merge,
)


def test_streaming_telemetry_anomaly_detection_pipeline():
    """Verify end-to-end anomaly detection isolates injected anomalous channels under FDR control."""
    results = run_streaming_telemetry_benchmark()

    # Channels 4 and 5 were injected with mean and variance shift anomalies
    anomalous = results["anomalous_channels"]
    assert 4 in anomalous, f"Channel 4 was not detected as anomalous: {anomalous}"
    assert 5 in anomalous, f"Channel 5 was not detected as anomalous: {anomalous}"

    # Channels 0, 1, 2, 3 should remain non-anomalous
    for ch in range(4):
        assert ch not in anomalous, f"False positive detected on nominal channel {ch}"

    # Verify moments were updated online for all 6 channels
    for ch in range(6):
        cnt, mean, m2, m3, m4 = results["moments"][ch]
        assert cnt == 60, f"Expected 60 streaming samples, got {cnt}"
        assert m2 > 0.0, f"Second moment m2 should be strictly positive: {m2}"


def test_jensen_shannon_metric_properties():
    """Verify that JS metric distance satisfies identity and non-negativity."""
    p = [0.1, 0.4, 0.5]
    q = [0.1, 0.4, 0.5]
    dist_zero = jensen_shannon_metric_distance(p, q)
    assert abs(dist_zero) < 1e-12, f"Expected 0 distance for identical distributions, got {dist_zero}"

    r = [0.8, 0.1, 0.1]
    dist_pos = jensen_shannon_metric_distance(p, r)
    assert dist_pos > 0.1, f"Expected positive distance, got {dist_pos}"

    # Symmetry
    dist_sym = jensen_shannon_metric_distance(r, p)
    assert abs(dist_pos - dist_sym) < 1e-12, "JS metric distance must be symmetric"


def test_welford_online_moments_accuracy():
    """Verify online Welford moments match two-pass batch calculations within 1e-12."""
    import random
    rng = random.Random(1337)
    data = [rng.gauss(5.0, 2.0) for _ in range(100)]

    cnt = 0
    mean = 0.0
    m2 = 0.0
    m3 = 0.0
    m4 = 0.0

    for x in data:
        cnt, mean, m2, m3, m4 = welford_skewness_kurtosis(cnt, mean, m2, m3, m4, x)

    # Batch comparison
    n = len(data)
    batch_mean = sum(data) / n
    batch_m2 = sum((x - batch_mean) ** 2 for x in data)

    assert abs(mean - batch_mean) < 1e-12, f"Mean mismatch: {mean} vs {batch_mean}"
    assert abs(m2 - batch_m2) < 1e-10, f"Variance m2 mismatch: {m2} vs {batch_m2}"


def test_t_digest_weight_conservation():
    """Verify that t-digest centroid merging preserves total sample weight."""
    means = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    weights = [2.0, 1.0, 4.0, 3.0, 2.0, 5.0, 1.0, 2.0, 3.0, 4.0]
    total_w_in = sum(weights)

    out_m, out_w = t_digest_centroid_merge(means, weights, max_centroids=4)
    total_w_out = sum(out_w)

    assert abs(total_w_in - total_w_out) < 1e-12, f"Weight not conserved: {total_w_in} vs {total_w_out}"
    assert len(out_m) <= 6, f"Expected <= 6 centroids after merging, got {len(out_m)}"


def test_fdr_control_filter():
    """Verify Benjamini-Hochberg FDR step rejects only hypotheses meeting rank threshold."""
    p_values = [0.001, 0.005, 0.02, 0.30, 0.80]
    rejections = benjamini_hochberg_fdr_step(p_values, fdr_q=0.05)

    assert rejections[0] is True
    assert rejections[1] is True
    assert rejections[3] is False
    assert rejections[4] is False


def test_composed_concepts_registered_in_catalog():
    """Verify that all 6 composed building blocks exist in statistics_and_information.json."""
    cat_path = Path(__file__).resolve().parent.parent.parent.parent / "data" / "domains" / "statistics_and_information.json"
    assert cat_path.is_file(), f"Missing catalog file at {cat_path}"
    with open(cat_path, "r", encoding="utf-8") as f:
        concepts = json.load(f)

    catalog_names = {c["name"] for c in concepts}
    required = {
        "welford_skewness_kurtosis",
        "t_digest_centroid_merge",
        "median_absolute_deviation_mad",
        "kullback_leibler_discrete_div",
        "jensen_shannon_metric_distance",
        "benjamini_hochberg_fdr_step",
    }
    assert required.issubset(catalog_names), f"Missing concepts: {required - catalog_names}"
