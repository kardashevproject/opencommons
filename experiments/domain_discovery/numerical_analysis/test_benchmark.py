"""Pytest test suite for Numerical Analysis Stiff BVP Benchmark."""

import json
from pathlib import Path
import sys
import pytest

repo_root = Path(__file__).resolve().parent.parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from experiments.domain_discovery.numerical_analysis.benchmark import (
    solve_stiff_bvp_spectral_collocation,
    compute_multigrid_richardson_convergence,
    barycentric_lagrange_interpolate,
    chebyshev_nodes_1d,
)


def test_bvp_spectral_collocation_convergence():
    """Verify that Chebyshev collocation solver converges on non-linear stiff BVP within tolerance."""
    res = solve_stiff_bvp_spectral_collocation(n_nodes=33, epsilon=0.05, max_iters=25, tol=1e-7)

    assert res["final_residual"] < 1e-7, f"Residual too high: {res['final_residual']}"
    assert res["iterations"] < 25, f"Did not converge in 25 iterations: {res['iterations']}"

    # Verify boundary conditions u(-1) = 0, u(1) = 0
    assert abs(res["u"][0]) < 1e-12, f"Left boundary violated: {res['u'][0]}"
    assert abs(res["u"][-1]) < 1e-12, f"Right boundary violated: {res['u'][-1]}"

    # Pointwise accuracy against exact solution
    assert res["pointwise_error"] < 1e-3, f"Pointwise error too high: {res['pointwise_error']}"


def test_bvp_integral_conservation():
    """Verify that Simpson quadrature of reconstructed field matches analytical integral."""
    res = solve_stiff_bvp_spectral_collocation(n_nodes=33, epsilon=0.05)

    assert res["conservation_error"] < 1e-3, f"Conservation error too high: {res['conservation_error']}"
    assert res["integral_u"] > 0.0, "Integral should be positive for positive solution profile"


def test_multigrid_richardson_extrapolation():
    """Verify Richardson extrapolation cancels leading order error across grid levels."""
    rg = compute_multigrid_richardson_convergence()

    assert rg["err_fine"] < rg["err_coarse"], "Fine grid error should be less than coarse"
    assert rg["err_extrapolated"] < rg["err_fine"], "Extrapolated error should beat fine grid error"
    assert rg["convergence_rate"] > 3.0, f"Expected 2nd-order convergence rate ~4, got {rg['convergence_rate']}"


def test_barycentric_interpolation_knot_property():
    """Verify exact recovery of node values at Chebyshev collocation points."""
    nodes = chebyshev_nodes_1d(17, -1.0, 1.0)
    ys = [math_sin(x) for x in nodes]
    n = len(nodes)
    ws = [0.5 if (i == 0 or i == n - 1) else 1.0 for i in range(n)]
    for i in range(1, n, 2):
        ws[i] = -ws[i]

    for i, x in enumerate(nodes):
        y_eval = barycentric_lagrange_interpolate(nodes, ys, ws, x)
        assert abs(y_eval - ys[i]) < 1e-12, f"Knot property failed at index {i}: {y_eval} vs {ys[i]}"


def math_sin(x: float) -> float:
    import math
    return math.sin(x)


def test_composed_concepts_registered_in_catalog():
    """Verify that all 6 composed building blocks exist in numerical_analysis.json."""
    cat_path = Path(__file__).resolve().parent.parent.parent.parent / "data" / "domains" / "numerical_analysis.json"
    assert cat_path.is_file()
    with open(cat_path, "r", encoding="utf-8") as f:
        concepts = json.load(f)

    catalog_names = {c["name"] for c in concepts}
    required = {
        "chebyshev_nodes_1d",
        "tridiagonal_thomas_solve",
        "brent_dekker_root",
        "barycentric_lagrange_interpolate",
        "simpson_adaptive_quad",
        "richardson_extrapolation_step",
    }
    assert required.issubset(catalog_names), f"Missing concepts: {required - catalog_names}"
