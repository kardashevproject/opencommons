"""Unit tests for Multi-Asset Risk Engine & Vol Surface Benchmark."""

from __future__ import annotations

import math
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from experiments.domain_discovery.quantitative_finance.benchmark import (
    FUNCS,
    run_multi_asset_risk_benchmark,
)


def test_default_multi_asset_risk_benchmark_passes():
    res = run_multi_asset_risk_benchmark(
        spot_price=100.0,
        rate=0.04,
        time_to_maturity=1.0,
        base_sigma=0.20,
    )
    assert res.passed is True
    assert len(res.asset_volatilities) == 3
    assert abs(sum(res.portfolio_weights) - 1.0) < 1e-5
    assert res.portfolio_variance > 0.0
    assert res.cvar_95 >= res.var_95
    assert res.max_drawdown > 0.0
    assert res.put_call_parity_max_error < 1e-4
    assert 0.4 < res.delta_call < 0.7
    assert res.gamma > 0.0
    assert res.vega > 0.0


def test_risk_parity_weights_and_markowitz_variance():
    inv_w = FUNCS["inverse_variance_weights"]
    mark_var = FUNCS["markowitz_portfolio_variance"]
    variances = [0.04, 0.09, 0.16]
    w = inv_w(variances)
    assert abs(sum(w) - 1.0) < 1e-6
    assert w[0] > w[1] > w[2]  # lower variance gets higher weight
    cov = [
        [0.04, 0.01, 0.005],
        [0.01, 0.09, 0.02],
        [0.005, 0.02, 0.16],
    ]
    p_var = mark_var(w, cov)
    assert p_var > 0.0
    assert p_var < max(variances)


def test_put_call_parity_exact():
    call_fn = FUNCS["black_scholes_call_price"]
    put_fn = FUNCS["black_scholes_put_price"]
    spot = 120.0
    rate = 0.05
    t = 0.5
    vol = 0.25
    discount = math.exp(-rate * t)
    for strike in [90.0, 100.0, 110.0, 120.0, 130.0]:
        c = call_fn(spot, strike, t, rate, vol)
        p = put_fn(spot, strike, t, rate, vol)
        lhs = c - p
        rhs = spot - strike * discount
        assert abs(lhs - rhs) < 1e-4


def test_greeks_properties():
    delta_fn = FUNCS["black_scholes_delta_call"]
    gamma_fn = FUNCS["black_scholes_gamma"]
    vega_fn = FUNCS["black_scholes_vega"]
    spot = 100.0
    rate = 0.03
    t = 1.0
    vol = 0.20
    # ITM call should have higher delta than OTM call
    d_itm = delta_fn(spot, 80.0, t, rate, vol)
    d_atm = delta_fn(spot, 100.0, t, rate, vol)
    d_otm = delta_fn(spot, 120.0, t, rate, vol)
    assert d_itm > d_atm > d_otm
    assert 0.0 <= d_otm and d_itm <= 1.0
    # Gamma should peak around ATM
    g_atm = gamma_fn(spot, 100.0, t, rate, vol)
    g_otm = gamma_fn(spot, 130.0, t, rate, vol)
    assert g_atm > g_otm
    assert vega_fn(spot, 100.0, t, rate, vol) > 0.0


def test_cvar_subadditivity_and_drawdown():
    var_fn = FUNCS["value_at_risk_historical"]
    cvar_fn = FUNCS["conditional_value_at_risk_cvar"]
    mdd_fn = FUNCS["maximum_drawdown_percentage"]
    rets = [-0.08, -0.05, -0.03, -0.01, 0.0, 0.01, 0.02, 0.04, 0.05, 0.07]
    var_val = var_fn(rets, 0.90)
    cvar_val = cvar_fn(rets, 0.90)
    assert cvar_val >= var_val
    prices = [100.0, 120.0, 90.0, 110.0, 80.0, 100.0]
    # Peak is 120.0, trough is 80.0 -> drop is (120 - 80) / 120 = 40 / 120 = 0.33333
    mdd = mdd_fn(prices)
    assert abs(mdd - (120.0 - 80.0) / 120.0) < 1e-4
