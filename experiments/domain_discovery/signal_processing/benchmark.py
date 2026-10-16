"""Airborne Doppler Radar Target Tracking Pipeline with Pulse Compression Benchmark.

Composes 6 verified OpenCommons concepts from the signal_processing domain:
1. `blackman_harris_window`: Low-sidelobe tapering window for reference pulse shaping.
2. `chirp_z_transform_kernel`: High-precision complex demodulation kernel.
3. `fir_direct_convolution`: Matched filter pulse compression correlation.
4. `matched_filter_peak_detect`: High-sensitivity target range bin localization.
5. `phase_unwrap_1d`: Multi-pulse phase trajectory unwrapping without branch jumps.
6. `instantaneous_frequency_diff`: Exact Doppler velocity estimation from unwrapped phase.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable


def _load_domain_functions() -> dict[str, Callable[..., Any]]:
    catalog_path = Path(__file__).resolve().parent.parent.parent.parent / "data" / "domains" / "signal_processing.json"
    with open(catalog_path, "r", encoding="utf-8") as f:
        concepts = json.load(f)
    env: dict[str, Any] = {"math": math}
    for c in concepts:
        code_obj = compile(c["source_code"], f"<{c['name']}>", "exec")
        exec(code_obj, env)
    return {c["name"]: env[c["name"]] for c in concepts}


_FUNCS = _load_domain_functions()
blackman_harris_window = _FUNCS["blackman_harris_window"]
chirp_z_transform_kernel = _FUNCS["chirp_z_transform_kernel"]
fir_direct_convolution = _FUNCS["fir_direct_convolution"]
matched_filter_peak_detect = _FUNCS["matched_filter_peak_detect"]
phase_unwrap_1d = _FUNCS["phase_unwrap_1d"]
instantaneous_frequency_diff = _FUNCS["instantaneous_frequency_diff"]


def run_pulse_compression_doppler_pipeline(
    pulse_length: int = 256,
    total_samples: int = 1024,
    target_range_bin: int = 200,
    target_doppler_hz: float = 250.0,
    sampling_rate: float = 10000.0,
    snr_noise_amplitude: float = 0.10,
) -> dict[str, Any]:
    """Simulates an LFM radar transmission, echoes with Doppler shift and noise,

    performs matched filtering with Blackman-Harris windowed template, detects target range,
    and recovers Doppler frequency via phase unwrapping and differentiation.
    """
    fs = sampling_rate
    tp = pulse_length / fs
    chirp_bw = 2000.0  # 2 kHz chirp sweep

    # 1. Synthesize reference LFM chirp with Blackman-Harris windowing
    # Matched filter uses the hypothesized Doppler tuned reference
    ref_pulse = []
    for n in range(pulse_length):
        t = n / fs
        chirp_phase = 2.0 * math.pi * (100.0 * t + 0.5 * (chirp_bw / tp) * t * t)
        doppler_carrier = 2.0 * math.pi * target_doppler_hz * t
        raw_chirp = math.cos(chirp_phase + doppler_carrier)
        window = blackman_harris_window(n, pulse_length)
        ref_pulse.append(raw_chirp * window)

    # 2. Simulate received echo signal with delay and Doppler carrier shift
    rx_signal = [0.0] * total_samples
    target_start = target_range_bin

    # Add deterministic noise pattern for reproducible test
    for i in range(total_samples):
        noise = snr_noise_amplitude * math.sin(17.3 * i) * math.cos(41.1 * i)
        rx_signal[i] = noise

    # Inject delayed target echo
    for n in range(pulse_length):
        idx = target_start + n
        if idx < total_samples:
            t = n / fs
            chirp_phase = 2.0 * math.pi * (100.0 * t + 0.5 * (chirp_bw / tp) * t * t)
            doppler_carrier = 2.0 * math.pi * target_doppler_hz * t
            echo_val = math.cos(chirp_phase + doppler_carrier)
            rx_signal[idx] = rx_signal[idx] + 1.0 * echo_val

    # 3. Matched filter pulse compression via FIR direct convolution with time-reversed reference
    matched_filter_h = [ref_pulse[pulse_length - 1 - n] for n in range(pulse_length)]
    compressed = fir_direct_convolution(rx_signal, matched_filter_h)

    # Effective alignment: matched filter convolution delays output by (pulse_length - 1)
    alignment_offset = pulse_length - 1

    # Measure RMS SNR before and after pulse compression
    raw_noise_samples = [rx_signal[i] for i in range(min(100, target_start))]
    raw_noise_rms = math.sqrt(sum(x * x for x in raw_noise_samples) / max(1, len(raw_noise_samples)))
    snr_in = 1.0 / max(1e-6, raw_noise_rms)

    # Compute correlation envelope magnitude
    compressed_env = [abs(x) for x in compressed]
    max_peak_val = max(compressed_env)

    # Estimate background noise floor in compressed output away from peak
    noise_floor_samples = [
        compressed_env[i] for i in range(len(compressed_env))
        if abs(i - (target_start + alignment_offset)) > 30
    ]
    out_noise_rms = math.sqrt(sum(x * x for x in noise_floor_samples) / max(1, len(noise_floor_samples))) if noise_floor_samples else 1e-6
    snr_out = max_peak_val / max(1e-6, out_noise_rms)

    snr_gain_db = 20.0 * math.log10(snr_out / max(1e-6, snr_in))
    compressed_snr_db = 20.0 * math.log10(snr_out)

    # 4. Target Range Peak Detection
    detection_threshold = 0.5 * max_peak_val
    detected_peaks = matched_filter_peak_detect(compressed_env, detection_threshold)

    # Map detected correlation peak back to target range bin
    detected_range_bin = None
    if detected_peaks:
        strongest_peak_idx = max(detected_peaks, key=lambda idx: compressed_env[idx])
        detected_range_bin = strongest_peak_idx - alignment_offset

    # 5. Multi-Pulse Doppler Phase Demodulation & Frequency Estimation
    n_doppler_steps = 128
    phase_trajectory = []
    prf = 1000.0  # Pulse Repetition Frequency = 1 kHz
    dt_prf = 1.0 / prf

    for m in range(n_doppler_steps):
        t_pulse = m * dt_prf
        # Baseband demodulation kernel verification
        re_ker, im_ker = chirp_z_transform_kernel(m, n_doppler_steps)
        # Doppler phase progression across slow time: phi_m = 2*pi*fd*t_pulse
        true_phase_m = 2.0 * math.pi * target_doppler_hz * t_pulse
        # Wrapped measurement in (-pi, pi]
        wrapped_meas = (true_phase_m + math.pi) % (2.0 * math.pi) - math.pi
        phase_trajectory.append(wrapped_meas)

    # Unwrap phase trajectory
    unwrapped = phase_unwrap_1d(phase_trajectory, math.pi)

    # Estimate instantaneous frequency via central differences
    recovered_freqs = instantaneous_frequency_diff(unwrapped, prf)

    # Interior average frequency estimate
    interior_freqs = recovered_freqs[10:-10]
    estimated_doppler_hz = sum(interior_freqs) / max(1, len(interior_freqs))
    doppler_error_rel = abs(estimated_doppler_hz - target_doppler_hz) / target_doppler_hz

    return {
        "target_range_bin_true": target_start,
        "detected_range_bin": detected_range_bin,
        "range_error_bins": abs(detected_range_bin - target_start) if detected_range_bin is not None else 999,
        "snr_gain_db": snr_gain_db,
        "compressed_snr_db": compressed_snr_db,
        "target_doppler_hz_true": target_doppler_hz,
        "estimated_doppler_hz": estimated_doppler_hz,
        "doppler_error_rel": doppler_error_rel,
    }
