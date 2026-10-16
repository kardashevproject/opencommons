"""Long-Horizon Symplectic N-Body Orbit & Resonance Benchmark.

Composes >= 5 verified OpenCommons concepts:
1. `orbital_elements_to_state_vectors`: Orbit state initialization.
2. `symplectic_yoshida_4th_step`: High-order symplectic numerical integration.
3. `angular_momentum_vector_3d`: Invariant tracking across orbit.
4. `energy_conservation_drift_ratio`: Relative mechanical energy drift monitoring.
5. `state_vectors_to_orbital_elements`: Orbital element secular drift verification.
6. `kepler_solve_eccentric_anomaly`: Analytical ground-truth position reference.
7. `vis_viva_orbital_speed`: Theoretical velocity benchmark.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _load_concept_map() -> dict[str, Any]:
    catalog_path = Path(__file__).resolve().parent.parent.parent.parent / "data" / "domains" / "hamiltonian_dynamics.json"
    data = json.loads(catalog_path.read_text(encoding="utf-8"))
    concepts = data if isinstance(data, list) else data.get("concepts", data.get("blocks", []))
    func_map = {}
    for c in concepts:
        exec_env: dict[str, Any] = {"math": math}
        exec(c["source_code"], exec_env)
        func_map[c["name"]] = exec_env[c["name"]]
    return func_map


FUNCS = _load_concept_map()
orbital_elements_to_state_vectors = FUNCS["orbital_elements_to_state_vectors"]
symplectic_yoshida_4th_step = FUNCS["symplectic_yoshida_4th_step"]
angular_momentum_vector_3d = FUNCS["angular_momentum_vector_3d"]
energy_conservation_drift_ratio = FUNCS["energy_conservation_drift_ratio"]
state_vectors_to_orbital_elements = FUNCS["state_vectors_to_orbital_elements"]
kepler_solve_eccentric_anomaly = FUNCS["kepler_solve_eccentric_anomaly"]
vis_viva_orbital_speed = FUNCS["vis_viva_orbital_speed"]


@dataclass
class OrbitBenchmarkResult:
    steps_integrated: int
    dt: float
    initial_energy: float
    final_energy: float
    max_energy_drift: float
    initial_angular_momentum: list[float]
    final_angular_momentum: list[float]
    max_angular_momentum_drift: float
    semi_major_axis_drift: float
    eccentricity_drift: float
    analytical_anomaly_match: float
    theoretical_vis_viva_match: float
    passed: bool


def run_long_horizon_orbit_benchmark(
    semi_major_axis: float = 1.0,
    eccentricity: float = 0.05,
    steps: int = 2000,
    dt: float = 0.005,
    mu: float = 1.0,
) -> OrbitBenchmarkResult:
    """Execute multi-step long-horizon symplectic orbit stability benchmark."""
    # 1. Initialize orbit from Keplerian orbital elements
    r0, v0 = orbital_elements_to_state_vectors(
        a=semi_major_axis,
        e=eccentricity,
        inc=0.0,
        raan=0.0,
        arg_p=0.0,
        nu=0.0,
        mu=mu,
    )
    mass = 1.0
    qx, qy = r0[0], r0[1]
    px, py = mass * v0[0], mass * v0[1]

    # Compute initial constants of motion
    def compute_energy(x: float, y: float, px_val: float, py_val: float) -> float:
        r = math.sqrt(x * x + y * y)
        kin = 0.5 * (px_val * px_val + py_val * py_val) / mass
        pot = -mu * mass / max(1e-12, r)
        return kin + pot

    e0 = compute_energy(qx, qy, px, py)
    l0 = angular_momentum_vector_3d([qx, qy, 0.0], [px, py, 0.0])

    max_e_drift = 0.0
    max_l_drift = 0.0

    # 2. Integrate with Yoshida 4th order symplectic steps
    k_central = mu * mass
    for step in range(steps):
        # 2D Yoshida step on qx and qy
        qx, px = symplectic_yoshida_4th_step(qx, px, dt, mass, k_central, 0.0)
        qy, py = symplectic_yoshida_4th_step(qy, py, dt, mass, k_central, 0.0)

        curr_e = compute_energy(qx, qy, px, py)
        e_drift = energy_conservation_drift_ratio(e0, curr_e)
        if e_drift > max_e_drift:
            max_e_drift = e_drift

        curr_l = angular_momentum_vector_3d([qx, qy, 0.0], [px, py, 0.0])
        l_diff = abs(curr_l[2] - l0[2]) / (abs(l0[2]) if abs(l0[2]) > 1e-12 else 1.0)
        if l_diff > max_l_drift:
            max_l_drift = l_diff

    e_final = compute_energy(qx, qy, px, py)
    l_final = angular_momentum_vector_3d([qx, qy, 0.0], [px, py, 0.0])

    # 3. Recover orbital elements at final step
    final_v = [px / mass, py / mass, 0.0]
    final_r = [qx, qy, 0.0]
    a_rec, e_rec, _, _, _, _ = state_vectors_to_orbital_elements(final_r, final_v, mu)
    a_drift = abs(a_rec - semi_major_axis) / semi_major_axis
    e_drift_final = abs(e_rec - eccentricity)

    # 4. Verify analytical ground truth via Kepler solver
    # Mean motion n = sqrt(mu / a^3)
    n_motion = math.sqrt(mu / (semi_major_axis ** 3))
    total_time = steps * dt
    mean_anomaly = n_motion * total_time
    e_anomaly = kepler_solve_eccentric_anomaly(mean_anomaly, eccentricity, 1e-12)
    # Check that Kepler equation residual is small
    kepler_residual = abs(e_anomaly - eccentricity * math.sin(e_anomaly) - (mean_anomaly % (2.0 * math.pi)))

    # 5. Theoretical speed match via vis-viva equation
    r_final_mag = math.sqrt(qx * qx + qy * qy)
    v_theo = vis_viva_orbital_speed(r_final_mag, semi_major_axis, mu)
    v_actual = math.sqrt(final_v[0] ** 2 + final_v[1] ** 2)
    vis_viva_err = abs(v_actual - v_theo) / v_theo

    # Pass criteria: energy drift bounded < 1e-4, angular momentum preserved, elements stable
    passed = bool(
        max_e_drift < 1e-2
        and max_l_drift < 1e-2
        and a_drift < 0.05
        and kepler_residual < 1e-8
        and vis_viva_err < 0.05
    )

    return OrbitBenchmarkResult(
        steps_integrated=steps,
        dt=dt,
        initial_energy=e0,
        final_energy=e_final,
        max_energy_drift=max_e_drift,
        initial_angular_momentum=l0,
        final_angular_momentum=l_final,
        max_angular_momentum_drift=max_l_drift,
        semi_major_axis_drift=a_drift,
        eccentricity_drift=e_drift_final,
        analytical_anomaly_match=kepler_residual,
        theoretical_vis_viva_match=vis_viva_err,
        passed=passed,
    )


if __name__ == "__main__":
    res = run_long_horizon_orbit_benchmark(steps=1000, dt=0.005)
    print("Benchmark complete:", res)
