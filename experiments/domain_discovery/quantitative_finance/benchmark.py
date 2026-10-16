"""Multi-Asset Risk Engine & Vol Surface Benchmark for Quantitative Finance.

Composes >= 5 verified OpenCommons concepts:
1. `garman_klass_volatility`: OHLC historical volatility estimation for multi-asset universe.
2. `inverse_variance_weights`: Risk parity allocation weighting.
3. `markowitz_portfolio_variance`: Analytical quadratic portfolio risk evaluation w^T Sigma w.
4. `value_at_risk_historical`: Non-parametric historical simulation Value-at-Risk.
5. `conditional_value_at_risk_cvar`: Subadditive Expected Shortfall tail risk measurement.
6. `maximum_drawdown_percentage`: Path-dependent peak-to-trough capital preservation check.
7. `sharpe_ratio_annualized`: Annualized risk-adjusted return ratio.
8. `black_scholes_call_price`: Analytic option pricing across vol surface grid.
9. `black_scholes_put_price`: Put option pricing and Put-Call parity arbitrage verification.
10. `implied_volatility_newton`: Vol surface Newton-Raphson inversion.
11. `black_scholes_delta_call`: First-order directional hedging exposure.
12. `black_scholes_gamma`: Second-order curvature risk.
13. `black_scholes_vega`: Volatility sensitivity risk.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_repo_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))


def _load_concept_map() -> dict[str, Any]:
    catalog_path = Path(__file__).resolve().parent.parent.parent.parent / "data" / "domains" / "quantitative_finance.json"
    data = json.loads(catalog_path.read_text(encoding="utf-8"))
    concepts = data if isinstance(data, list) else data.get("concepts", data.get("blocks", []))
    func_map = {}
    for c in concepts:
        exec_env: dict[str, Any] = {"math": math}
        exec(c["source_code"], exec_env)
        func_map[c["name"]] = exec_env[c["name"]]
    return func_map


FUNCS = _load_concept_map()
garman_klass_volatility = FUNCS["garman_klass_volatility"]
inverse_variance_weights = FUNCS["inverse_variance_weights"]
markowitz_portfolio_variance = FUNCS["markowitz_portfolio_variance"]
value_at_risk_historical = FUNCS["value_at_risk_historical"]
conditional_value_at_risk_cvar = FUNCS["conditional_value_at_risk_cvar"]
maximum_drawdown_percentage = FUNCS["maximum_drawdown_percentage"]
sharpe_ratio_annualized = FUNCS["sharpe_ratio_annualized"]
black_scholes_call_price = FUNCS["black_scholes_call_price"]
black_scholes_put_price = FUNCS["black_scholes_put_price"]
implied_volatility_newton = FUNCS["implied_volatility_newton"]
black_scholes_delta_call = FUNCS["black_scholes_delta_call"]
black_scholes_gamma = FUNCS["black_scholes_gamma"]
black_scholes_vega = FUNCS["black_scholes_vega"]


@dataclass
class MultiAssetRiskEngineResult:
    asset_volatilities: list[float]
    portfolio_weights: list[float]
    portfolio_variance: float
    portfolio_annualized_volatility: float
    var_95: float
    cvar_95: float
    max_drawdown: float
    sharpe_ratio: float
    strikes: list[float]
    call_prices: list[float]
    put_prices: list[float]
    implied_volatilities: list[float]
    put_call_parity_max_error: float
    delta_call: float
    gamma: float
    vega: float
    passed: bool


def run_multi_asset_risk_benchmark(
    spot_price: float = 100.0,
    rate: float = 0.04,
    time_to_maturity: float = 1.0,
    base_sigma: float = 0.20,
) -> MultiAssetRiskEngineResult:
    """Execute end-to-end multi-asset risk engine and vol surface benchmark."""
    # 1. Multi-Asset Volatility Layer (3 synthetic assets from OHLC bars)
    asset_ohlc = [
        # Asset 1: Low vol equity
        (
            [100.0, 100.5, 101.0, 100.8, 101.2],
            [101.2, 101.5, 101.8, 101.6, 102.0],
            [99.5, 100.0, 100.4, 100.2, 100.8],
            [100.5, 101.0, 100.8, 101.2, 101.8],
        ),
        # Asset 2: Medium vol equity
        (
            [50.0, 51.0, 50.5, 52.0, 51.5],
            [52.0, 52.5, 53.0, 53.5, 54.0],
            [49.0, 50.0, 49.5, 50.5, 50.8],
            [51.0, 50.5, 52.0, 51.5, 53.0],
        ),
        # Asset 3: High vol commodity
        (
            [20.0, 21.0, 19.5, 22.0, 20.5],
            [22.5, 23.0, 22.0, 24.0, 23.5],
            [19.0, 19.5, 18.5, 20.0, 19.8],
            [21.0, 19.5, 22.0, 20.5, 22.8],
        ),
    ]

    vols = []
    for opens, highs, lows, closes in asset_ohlc:
        v = garman_klass_volatility(opens, highs, lows, closes)
        vols.append(v)

    variances = [v * v for v in vols]

    # 2. Inverse Variance Risk Parity Weights
    weights = inverse_variance_weights(variances)

    # 3. Markowitz Quadratic Portfolio Variance w^T Sigma w
    # Construct stylized covariance matrix with correlation rho = 0.3
    n_assets = len(vols)
    cov_matrix: list[list[float]] = []
    for i in range(n_assets):
        row = []
        for j in range(n_assets):
            corr = 1.0 if i == j else 0.3
            cov_ij = corr * vols[i] * vols[j]
            row.append(cov_ij)
        cov_matrix.append(row)

    port_var = markowitz_portfolio_variance(weights, cov_matrix)
    port_vol = math.sqrt(port_var)

    # 4. Portfolio Return Distribution & Downside Tail Risk
    # Synthetic multi-asset portfolio daily returns
    sim_returns = [
        -0.035, -0.022, -0.015, -0.008, -0.004,
        0.001, 0.005, 0.008, 0.012, 0.015,
        0.018, 0.021, 0.025, 0.028, 0.032,
    ]
    var_95 = value_at_risk_historical(sim_returns, 0.90)
    cvar_95 = conditional_value_at_risk_cvar(sim_returns, 0.90)

    # Synthetic cumulative price path for drawdown & Sharpe
    price_path = [100.0, 102.5, 105.0, 98.0, 96.5, 101.0, 104.0, 108.5]
    max_dd = maximum_drawdown_percentage(price_path)
    sharpe = sharpe_ratio_annualized(sim_returns, risk_free_rate=rate, periods_per_year=252)

    # 5. Volatility Surface & Derivatives Layer
    # Strikes across moneyness K = [80, 90, 100, 110, 120]
    strikes = [0.8 * spot_price, 0.9 * spot_price, 1.0 * spot_price, 1.1 * spot_price, 1.2 * spot_price]
    call_prices = []
    put_prices = []
    iv_calibrated = []
    parity_errors = []

    discount = math.exp(-rate * time_to_maturity)

    for k in strikes:
        # Volatility smile: sigma(K) = base_sigma + 0.1 * (ln(K/S))^2
        moneyness_log = math.log(k / spot_price)
        vol_k = base_sigma + 0.1 * (moneyness_log * moneyness_log)

        c = black_scholes_call_price(spot_price, k, time_to_maturity, rate, vol_k)
        p = black_scholes_put_price(spot_price, k, time_to_maturity, rate, vol_k)

        # Put-Call parity check: Call - Put == Spot - Strike * exp(-r*T)
        parity_lhs = c - p
        parity_rhs = spot_price - k * discount
        parity_errors.append(abs(parity_lhs - parity_rhs))

        # Invert call price for implied vol
        iv = implied_volatility_newton(c, spot_price, k, time_to_maturity, rate, max_iter=30)
        call_prices.append(c)
        put_prices.append(p)
        iv_calibrated.append(iv)

    max_parity_error = max(parity_errors)

    # 6. At-the-money Greeks
    delta_atm = black_scholes_delta_call(spot_price, spot_price, time_to_maturity, rate, base_sigma)
    gamma_atm = black_scholes_gamma(spot_price, spot_price, time_to_maturity, rate, base_sigma)
    vega_atm = black_scholes_vega(spot_price, spot_price, time_to_maturity, rate, base_sigma)

    all_passed = (
        len(vols) == 3
        and all(v > 0.0 for v in vols)
        and abs(sum(weights) - 1.0) < 1e-5
        and port_var > 0.0
        and cvar_95 >= var_95
        and max_dd > 0.0
        and max_parity_error < 1e-4
        and (0.4 < delta_atm < 0.7)
        and gamma_atm > 0.0
        and vega_atm > 0.0
    )

    return MultiAssetRiskEngineResult(
        asset_volatilities=vols,
        portfolio_weights=weights,
        portfolio_variance=port_var,
        portfolio_annualized_volatility=port_vol,
        var_95=var_95,
        cvar_95=cvar_95,
        max_drawdown=max_dd,
        sharpe_ratio=sharpe,
        strikes=strikes,
        call_prices=call_prices,
        put_prices=put_prices,
        implied_volatilities=iv_calibrated,
        put_call_parity_max_error=max_parity_error,
        delta_call=delta_atm,
        gamma=gamma_atm,
        vega=vega_atm,
        passed=all_passed,
    )
