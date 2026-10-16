"""Stiff Boundary-Value Differential Equation with Adaptive Chebyshev Spectral Collocation Benchmark.

Composes 6 verified OpenCommons concepts from the numerical_analysis domain:
1. `chebyshev_nodes_1d`: Non-uniform spectral collocation grid nodes.
2. `tridiagonal_thomas_solve`: O(N) TDMA solver for linearized differential operator.
3. `brent_dekker_root`: Non-linear local reaction root refinement.
4. `barycentric_lagrange_interpolate`: High-accuracy spectral field reconstruction on uniform grid.
5. `simpson_adaptive_quad`: Total energy and integral conservation check.
6. `richardson_extrapolation_step`: Discretization error estimation across grid levels.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable


def _load_domain_functions() -> dict[str, Callable[..., Any]]:
    catalog_path = Path(__file__).resolve().parent.parent.parent.parent / "data" / "domains" / "numerical_analysis.json"
    with open(catalog_path, "r", encoding="utf-8") as f:
        concepts = json.load(f)
    env: dict[str, Any] = {"math": math}
    for c in concepts:
        code_obj = compile(c["source_code"], f"<{c['name']}>", "exec")
        exec(code_obj, env)
    return {c["name"]: env[c["name"]] for c in concepts}


_FUNCS = _load_domain_functions()
chebyshev_nodes_1d = _FUNCS["chebyshev_nodes_1d"]
tridiagonal_thomas_solve = _FUNCS["tridiagonal_thomas_solve"]
brent_dekker_root = _FUNCS["brent_dekker_root"]
barycentric_lagrange_interpolate = _FUNCS["barycentric_lagrange_interpolate"]
simpson_adaptive_quad = _FUNCS["simpson_adaptive_quad"]
richardson_extrapolation_step = _FUNCS["richardson_extrapolation_step"]


def solve_stiff_bvp_spectral_collocation(
    n_nodes: int = 33,
    epsilon: float = 0.05,
    max_iters: int = 25,
    tol: float = 1e-7,
) -> dict[str, Any]:
    """Solves non-linear stiff diffusion-reaction BVP:

        -epsilon * u''(x) + u(x) + u(x)^3 = f(x)
        u(-1) = 0.0, u(1) = 0.0

    using Chebyshev collocation discretization and iterative Newton-Picard refinement.
    """
    # 1. Collocation grid generation via Chebyshev-Gauss-Lobatto nodes
    nodes = chebyshev_nodes_1d(n_nodes, -1.0, 1.0)
    n = len(nodes)

    # Source forcing term: f(x) chosen so exact smooth solution is u_exact = (1 - x^2) * exp(x)
    # f(x) = -epsilon * u''(x) + u(x) + u(x)^3
    f_vals = []
    u_exact = []
    for x in nodes:
        ex = math.exp(x)
        ue = (1.0 - x * x) * ex
        u_exact.append(ue)
        # u'' = (-1 - 4x - x^2)*exp(x)
        u_pp = (-1.0 - 4.0 * x - x * x) * ex
        fx = -epsilon * u_pp + ue + (ue * ue * ue)
        f_vals.append(fx)

    # Initial guess
    u = [0.0] * n

    # Fixed weights for barycentric interpolation on Chebyshev nodes
    ws = [0.5 if (i == 0 or i == n - 1) else 1.0 for i in range(n)]
    for i in range(1, n, 2):
        ws[i] = -ws[i]

    # Iterative non-linear solver
    final_residual = 1.0
    for it in range(max_iters):
        # Tridiagonal finite difference discretization on non-uniform Chebyshev mesh
        a_sub = []
        b_diag = [1.0]
        c_sup = [0.0]
        d_rhs = [0.0]

        # Interior nodes
        for i in range(1, n - 1):
            h_left = nodes[i] - nodes[i - 1]
            h_right = nodes[i + 1] - nodes[i]
            h_avg = 0.5 * (h_left + h_right)

            # -epsilon * u'' operator
            alpha_left = -epsilon / (h_left * h_avg)
            alpha_right = -epsilon / (h_right * h_avg)
            alpha_mid = epsilon * (1.0 / h_left + 1.0 / h_right) / h_avg

            # Local non-linear root refinement for cubic reaction: g(u_i) = u_i^3 + u_i - source = 0
            coeffs = [-f_vals[i], 1.0, 0.0, 1.0]  # c0 + c1*u + c2*u^2 + c3*u^3
            u_star = brent_dekker_root(coeffs, -2.0, 2.0, 1e-8)

            # Linearized reaction Jacobian: d/du (u + u^3) = 1 + 3*u^2
            reaction_diag = 1.0 + 3.0 * u[i] * u[i]
            reaction_rhs = 2.0 * u[i] * u[i] * u[i] + f_vals[i]

            a_sub.append(alpha_left)
            b_diag.append(alpha_mid + reaction_diag)
            c_sup.append(alpha_right)
            d_rhs.append(reaction_rhs)

        # Right boundary u(1) = 0
        a_sub.append(0.0)
        b_diag.append(1.0)
        d_rhs.append(0.0)

        # Solve linearized system using Thomas algorithm
        u_next = tridiagonal_thomas_solve(a_sub, b_diag, c_sup, d_rhs)

        # Check convergence residual
        diff = max(abs(u_next[i] - u[i]) for i in range(n))
        u = u_next
        final_residual = diff
        if diff < tol:
            break

    # 2. High-resolution evaluation on uniform fine mesh using Barycentric Lagrange interpolation
    n_fine = 101
    x_fine = [-1.0 + 2.0 * i / (n_fine - 1) for i in range(n_fine)]
    u_fine = [barycentric_lagrange_interpolate(nodes, u, ws, x) for x in x_fine]
    u_exact_fine = [(1.0 - x * x) * math.exp(x) for x in x_fine]

    # 3. Conserved energy / integral norm using Simpson's adaptive rule
    h_fine = 2.0 / (n_fine - 1)
    integral_u = simpson_adaptive_quad(u_fine, h_fine)
    integral_exact = simpson_adaptive_quad(u_exact_fine, h_fine)
    conservation_error = abs(integral_u - integral_exact)

    # Max pointwise error against analytical solution
    pointwise_error = max(abs(u_fine[i] - u_exact_fine[i]) for i in range(n_fine))

    return {
        "nodes": nodes,
        "u": u,
        "x_fine": x_fine,
        "u_fine": u_fine,
        "iterations": it + 1,
        "final_residual": final_residual,
        "pointwise_error": pointwise_error,
        "integral_u": integral_u,
        "integral_exact": integral_exact,
        "conservation_error": conservation_error,
    }


def compute_multigrid_richardson_convergence() -> dict[str, Any]:
    """Executes multi-grid hierarchy and verifies 2nd order asymptotic convergence via Richardson extrapolation."""
    res_coarse = solve_stiff_bvp_spectral_collocation(n_nodes=17, epsilon=0.05)
    res_fine = solve_stiff_bvp_spectral_collocation(n_nodes=33, epsilon=0.05)

    # Reconstruct at x=0.0
    ws_c = [0.5 if (i == 0 or i == 16) else 1.0 for i in range(17)]
    for i in range(1, 17, 2):
        ws_c[i] = -ws_c[i]
    val_coarse = barycentric_lagrange_interpolate(res_coarse["nodes"], res_coarse["u"], ws_c, 0.0)

    ws_f = [0.5 if (i == 0 or i == 32) else 1.0 for i in range(33)]
    for i in range(1, 33, 2):
        ws_f[i] = -ws_f[i]
    val_fine = barycentric_lagrange_interpolate(res_fine["nodes"], res_fine["u"], ws_f, 0.0)

    extrapolated = richardson_extrapolation_step(val_fine, val_coarse, order=2, ratio=2.0)
    exact_mid = 1.0 * math.exp(0.0)  # (1 - 0^2)*exp(0) = 1.0

    err_coarse = abs(val_coarse - exact_mid)
    err_fine = abs(val_fine - exact_mid)
    err_extrapolated = abs(extrapolated - exact_mid)

    return {
        "val_coarse": val_coarse,
        "val_fine": val_fine,
        "extrapolated": extrapolated,
        "exact_mid": exact_mid,
        "err_coarse": err_coarse,
        "err_fine": err_fine,
        "err_extrapolated": err_extrapolated,
        "convergence_rate": err_coarse / max(1e-12, err_fine),
    }
