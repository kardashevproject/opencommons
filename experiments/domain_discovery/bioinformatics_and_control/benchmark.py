"""Metagenomic Sequence Assembly & Autonomous Closed-Loop Bioreactor Adaptive Control Benchmark.

Composes 7 verified OpenCommons concepts from the bioinformatics_and_control domain:
1. `kmer_frequency_table_dna`: Sequence composition analysis and k-mer profiling.
2. `burrows_wheeler_transform`: Metagenomic read indexing and bijective permutation.
3. `inverse_burrows_wheeler_transform`: Decompression and exact bit-for-bit sequence recovery.
4. `needleman_wunsch_global_align`: Pairwise global homology alignment against reference strain.
5. `kalman_filter_1d_update`: Real-time measurement noise reduction on thermal probes.
6. `pid_controller_discrete_step`: Proportional-Integral-Derivative temperature stabilization.
7. `pid_anti_windup_clamping`: Actuator power clamping and integrator freeze protection.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable


def _load_domain_functions() -> dict[str, Callable[..., Any]]:
    catalog_path = Path(__file__).resolve().parent.parent.parent.parent / "data" / "domains" / "bioinformatics_and_control.json"
    with open(catalog_path, "r", encoding="utf-8") as f:
        concepts = json.load(f)
    env: dict[str, Any] = {"math": math}
    for c in concepts:
        code_obj = compile(c["source_code"], f"<{c['name']}>", "exec")
        exec(code_obj, env)
    return {c["name"]: env[c["name"]] for c in concepts}


_FUNCS = _load_domain_functions()
kmer_frequency_table_dna = _FUNCS["kmer_frequency_table_dna"]
burrows_wheeler_transform = _FUNCS["burrows_wheeler_transform"]
inverse_burrows_wheeler_transform = _FUNCS["inverse_burrows_wheeler_transform"]
needleman_wunsch_global_align = _FUNCS["needleman_wunsch_global_align"]
kalman_filter_1d_update = _FUNCS["kalman_filter_1d_update"]
pid_controller_discrete_step = _FUNCS["pid_controller_discrete_step"]
pid_anti_windup_clamping = _FUNCS["pid_anti_windup_clamping"]


class MetagenomicPipeline:
    """Metagenomic read indexing, lossless compression, and alignment engine."""

    def analyze_read_composition(self, read_dna: str, k: int = 3) -> dict[str, int]:
        """Calculates k-mer frequency spectrum of genomic sequence."""
        return kmer_frequency_table_dna(read_dna, k)

    def compress_read_bwt(self, read_dna: str) -> tuple[str, int]:
        """Transforms genomic read into Burrows-Wheeler text and original index."""
        return burrows_wheeler_transform(read_dna)

    def decompress_read_bwt(self, bwt_text: str, original_idx: int) -> str:
        """Inverts BWT back to original genomic string with 100% bit-exact fidelity."""
        return inverse_burrows_wheeler_transform(bwt_text, original_idx)

    def align_to_reference(
        self,
        query_dna: str,
        reference_dna: str,
        match: int = 2,
        mismatch: int = -1,
        gap: int = -2,
    ) -> tuple[int, str, str]:
        """Aligns query read globally against reference genome strain."""
        return needleman_wunsch_global_align(query_dna, reference_dna, match, mismatch, gap)


class BioreactorThermalController:
    """Kalman-filtered PID closed-loop thermal bioreactor controller with anti-windup."""

    def __init__(
        self,
        target_temp: float = 37.0,
        kp: float = 4.0,
        ki: float = 0.4,
        kd: float = 1.2,
        min_power: float = -15.0,
        max_power: float = 15.0,
        initial_temp: float = 25.0,
    ) -> None:
        self.target_temp = float(target_temp)
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.min_power = float(min_power)
        self.max_power = float(max_power)

        # Kalman Filter State
        self.temp_est = float(initial_temp)
        self.p_cov = 1.0

        # PID Controller State
        self.prev_error = self.target_temp - self.temp_est
        self.integral = 0.0

    def step(self, raw_sensor_measurement: float, dt: float = 1.0) -> dict[str, float]:
        """Executes one filtering and control step."""
        # 1. Kalman measurement noise filtering
        # Measurement noise variance R = 0.05
        filtered_temp, new_p_cov, k_gain = kalman_filter_1d_update(
            self.temp_est, self.p_cov, float(raw_sensor_measurement), 0.05
        )
        self.temp_est = filtered_temp
        self.p_cov = new_p_cov + 0.01  # process noise increment

        # 2. PID control computation
        error = self.target_temp - self.temp_est
        raw_u, new_integral = pid_controller_discrete_step(
            error, self.prev_error, self.integral, self.kp, self.ki, self.kd, dt
        )

        # 3. Actuator saturation clamping & anti-windup freeze
        clamped_u, safe_integral = pid_anti_windup_clamping(
            raw_u, new_integral, error, self.min_power, self.max_power
        )

        self.integral = safe_integral
        self.prev_error = error

        return {
            "filtered_temp": self.temp_est,
            "control_power": clamped_u,
            "error": error,
            "kalman_gain": k_gain,
        }


def simulate_closed_loop_bioreactor(
    n_steps: int = 100,
    target_temp: float = 37.0,
    initial_temp: float = 25.0,
    ambient_temp: float = 22.0,
) -> dict[str, Any]:
    """Simulates physical bioreactor heating plant under Kalman-PID closed-loop control."""
    controller = BioreactorThermalController(
        target_temp=target_temp,
        initial_temp=initial_temp,
    )

    # Physical thermal plant state
    true_temp = float(initial_temp)
    dt = 0.5  # seconds

    temp_history: list[float] = [true_temp]
    control_history: list[float] = []

    for step_i in range(n_steps):
        # Deterministic pseudo-random measurement noise from sensor
        noise = 0.15 * math.sin(step_i * 1.7)
        noisy_meas = true_temp + noise

        # Controller execution
        ctrl_out = controller.step(noisy_meas, dt=dt)
        power = ctrl_out["control_power"]
        control_history.append(power)

        # 1st-order physical heat transfer ODE:
        # dT/dt = -k_loss * (T - T_ambient) + k_heater * Power
        k_loss = 0.08
        k_heater = 0.25
        dT = (-k_loss * (true_temp - ambient_temp) + k_heater * power) * dt
        true_temp += dT
        temp_history.append(true_temp)

    final_error = abs(true_temp - target_temp)
    return {
        "final_temp": true_temp,
        "target_temp": target_temp,
        "final_error": final_error,
        "settled": final_error <= 0.05,
        "temp_history": temp_history,
        "control_history": control_history,
    }
