"""Pytest test suite for Signal Processing Doppler Radar Benchmark."""

import json
import math
from pathlib import Path
import sys
import pytest

repo_root = Path(__file__).resolve().parent.parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from experiments.domain_discovery.signal_processing.benchmark import (
    run_pulse_compression_doppler_pipeline,
    blackman_harris_window,
    chirp_z_transform_kernel,
)


def test_pulse_compression_range_localization():
    """Verify that matched filter pulse compression localizes target range within +-1 bin."""
    res = run_pulse_compression_doppler_pipeline(
        pulse_length=64,
        total_samples=512,
        target_range_bin=180,
        target_doppler_hz=250.0,
    )

    assert res["detected_range_bin"] is not None, "Failed to detect target correlation peak"
    assert res["range_error_bins"] <= 1, f"Range error too large: {res['range_error_bins']} bins (true={res['target_range_bin_true']}, detected={res['detected_range_bin']})"


def test_pulse_compression_snr_gain():
    """Verify that pulse compression achieves at least 15 dB SNR improvement over raw signal."""
    res = run_pulse_compression_doppler_pipeline(
        pulse_length=256,
        total_samples=1024,
        target_range_bin=200,
        snr_noise_amplitude=0.10,
    )

    assert res["snr_gain_db"] >= 15.0, f"Insufficient SNR gain: {res['snr_gain_db']:.2f} dB (< 15 dB)"
    assert res["compressed_snr_db"] >= 25.0, f"Low compressed SNR: {res['compressed_snr_db']:.2f} dB"


def test_doppler_frequency_tracking_accuracy():
    """Verify that unwrapped phase differentiation estimates Doppler frequency within 0.05%."""
    target_hz = 325.0
    res = run_pulse_compression_doppler_pipeline(
        target_doppler_hz=target_hz,
    )

    assert res["doppler_error_rel"] < 0.0005, f"Doppler frequency error too high: {res['doppler_error_rel']*100:.4f}% (expected < 0.05%)"
    assert abs(res["estimated_doppler_hz"] - target_hz) < 0.2, f"Estimated {res['estimated_doppler_hz']} vs true {target_hz}"


def test_blackman_harris_window_properties():
    """Verify Blackman-Harris window peak, symmetry, and boundary attenuation."""
    n = 64
    w_start = blackman_harris_window(0, n)
    w_end = blackman_harris_window(n - 1, n)
    w_mid = blackman_harris_window(n // 2, n)

    # Low boundary attenuation
    assert w_start < 0.01, f"Boundary weight too large: {w_start}"
    assert w_end < 0.01, f"Boundary weight too large: {w_end}"

    # Peak near center
    assert w_mid > 0.95, f"Center peak too low: {w_mid}"

    # Symmetry
    for i in range(n // 2):
        w_left = blackman_harris_window(i, n)
        w_right = blackman_harris_window(n - 1 - i, n)
        assert abs(w_left - w_right) < 1e-12, f"Window asymmetry at index {i}"


def test_czt_kernel_unit_modulus():
    """Verify that Chirp Z-Transform kernel produces valid unit-modulus complex phasors."""
    for k in [0, 1, 5, 13, 31]:
        re, im = chirp_z_transform_kernel(k, 64)
        modulus_sq = re * re + im * im
        assert abs(modulus_sq - 1.0) < 1e-12, f"Modulus squared {modulus_sq} != 1.0 at k={k}"


def test_composed_concepts_registered_in_catalog():
    """Verify that all 6 composed building blocks exist in signal_processing.json."""
    cat_path = Path(__file__).resolve().parent.parent.parent.parent / "data" / "domains" / "signal_processing.json"
    assert cat_path.is_file()
    with open(cat_path, "r", encoding="utf-8") as f:
        concepts = json.load(f)

    catalog_names = {c["name"] for c in concepts}
    required = {
        "blackman_harris_window",
        "chirp_z_transform_kernel",
        "fir_direct_convolution",
        "matched_filter_peak_detect",
        "phase_unwrap_1d",
        "instantaneous_frequency_diff",
    }
    assert required.issubset(catalog_names), f"Missing concepts: {required - catalog_names}"
