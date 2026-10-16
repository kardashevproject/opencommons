"""Test suite for Hamiltonian Dynamics Long-Horizon Symplectic Orbit Benchmark."""

from __future__ import annotations

import math
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import pytest

from experiments.domain_discovery.hamiltonian_dynamics.benchmark import (
    run_long_horizon_orbit_benchmark,
    orbital_elements_to_state_vectors,
    symplectic_yoshida_4th_step,
    angular_momentum_vector_3d,
    energy_conservation_drift_ratio,
    state_vectors_to_orbital_elements,
    kepler_solve_eccentric_anomaly,
    vis_viva_orbital_speed,
)


def test_long_horizon_orbit_benchmark_execution():
    """Verify end-to-end execution of long-horizon orbit benchmark and passing status."""
    res = run_long_horizon_orbit_benchmark(
        semi_major_axis=1.0,
        eccentricity=0.04,
        steps=1000,
        dt=0.005,
        mu=1.0,
    )
    assert res.passed is True
    assert res.steps_integrated == 1000
    assert res.max_energy_drift < 1e-2
    assert res.max_angular_momentum_drift < 1e-2
    assert res.analytical_anomaly_match < 1e-8
    assert res.theoretical_vis_viva_match < 0.05


def test_symplectic_yoshida_composition():
    """Verify Yoshida 4th order integrator energy preservation on harmonic oscillator."""
    q, p = 1.0, 0.0
    m = 1.0
    k = 1.0
    dt = 0.01
    e0 = 0.5 * k * q * q + 0.5 * p * p / m
    for _ in range(500):
        q, p = symplectic_yoshida_4th_step(q, p, dt, m, k, 0.0)
    e_final = 0.5 * k * q * q + 0.5 * p * p / m
    drift = energy_conservation_drift_ratio(e0, e_final)
    assert drift < 1e-5


def test_kepler_solve_eccentric_anomaly_invariants():
    """Verify Kepler solver satisfies M = E - e*sin(E) across varied anomalies."""
    for e in (0.0, 0.1, 0.5, 0.9):
        for m in (0.0, 0.5, 1.5, 3.14, 5.0):
            e_anomaly = kepler_solve_eccentric_anomaly(m, e, 1e-12)
            res = abs(e_anomaly - e * math.sin(e_anomaly) - (m % (2.0 * math.pi)))
            assert res < 1e-10


def test_angular_momentum_conservation():
    """Verify angular momentum cross product is orthogonal to position and velocity."""
    r = [1.0, 2.0, 3.0]
    p = [-1.0, 0.5, 2.0]
    l = angular_momentum_vector_3d(r, p)
    # Dot products r . L and p . L must be zero
    dot_r = r[0] * l[0] + r[1] * l[1] + r[2] * l[2]
    dot_p = p[0] * l[0] + p[1] * l[1] + p[2] * l[2]
    assert abs(dot_r) < 1e-12
    assert abs(dot_p) < 1e-12


def test_vis_viva_consistency():
    """Verify vis-viva equation matches circular and parabolic theoretical benchmarks."""
    mu = 1.0
    # Circular orbit: r = a = 1.0 -> v = 1.0
    v_circ = vis_viva_orbital_speed(1.0, 1.0, mu)
    assert abs(v_circ - 1.0) < 1e-10

    # Parabolic escape: r = 1.0, a -> infinity -> v = sqrt(2)
    v_esc = vis_viva_orbital_speed(1.0, 1e9, mu)
    assert abs(v_esc - math.sqrt(2.0)) < 1e-5
