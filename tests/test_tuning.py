"""Equivalence tests ensuring vectorized MC tuning matches exact `run_backtest` strategies."""

import numpy as np

from etf_strategy.engine import run_backtest
from etf_strategy.metrics import summarize_backtest
from etf_strategy.synthetic import generate_gbm_paths
from etf_strategy.tuning import (
    _compute_rolling_std,
    _compute_simple_rsi,
    _compute_sma,
    _compute_trailing_dd_matrix,
    _compute_vol_ratio_matrix,
    _run_pacing_mult_mc,
    _run_signal_tranche_mc,
    _year_slices,
    build_strategy_from_params,
)


def test_vectorized_tuning_evaluators_match_exact_engine() -> None:
    paths = generate_gbm_paths(n_years=2, n_paths=3, annual_drift=0.05, annual_vol=0.22, seed=717)
    dates = paths[0].index
    prices = np.vstack([p["adj_close"].to_numpy(dtype=float) for p in paths])
    slices = _year_slices(dates)
    budget = 12_000.0

    cases = [
        (
            "below_sma",
            {"window": 20, "tranche_frac": 0.50},
            _run_signal_tranche_mc(
                prices,
                slices,
                (~np.isnan(_compute_sma(prices, 20))) & (prices < _compute_sma(prices, 20)),
                0.50,
                budget,
            )[0],
        ),
        (
            "rsi",
            {"window": 10, "threshold": 40.0, "tranche_frac": 0.50},
            _run_signal_tranche_mc(
                prices,
                slices,
                (~np.isnan(_compute_simple_rsi(prices, 10)))
                & (_compute_simple_rsi(prices, 10) <= 40.0),
                0.50,
                budget,
            )[0],
        ),
        (
            "bollinger_lower",
            {"window": 20, "num_std": 1.0, "tranche_frac": 0.50},
            _run_signal_tranche_mc(
                prices,
                slices,
                (~np.isnan(_compute_sma(prices, 20)))
                & (prices <= _compute_sma(prices, 20) - 1.0 * _compute_rolling_std(prices, 20)),
                0.50,
                budget,
            )[0],
        ),
        (
            "vol_scaled_pacing",
            {"short_window": 10, "long_window": 126, "sensitivity": 2.5},
            _run_pacing_mult_mc(
                prices,
                slices,
                np.clip(np.power(_compute_vol_ratio_matrix(prices, 10, 126), 2.5), 0.25, 4.0),
                budget,
            )[0],
        ),
        (
            "adaptive_dd_pacing",
            {"window": 63, "boost_per_10pct_dd": 8.0},
            _run_pacing_mult_mc(
                prices,
                slices,
                1.0 + 8.0 * (np.abs(_compute_trailing_dd_matrix(prices, 63)) / 0.10),
                budget,
            )[0],
        ),
    ]

    for family, params, vec_term in cases:
        fn = build_strategy_from_params(family, params)
        exact_term = np.array(
            [
                summarize_backtest(run_backtest(df, fn, strategy_name=family))["terminal_value"]
                for df in paths
            ],
            dtype=float,
        )
        np.testing.assert_allclose(
            vec_term,
            exact_term,
            rtol=1e-9,
            atol=1e-6,
            err_msg=f"Tuning vectorized mismatch for {family} ({params})",
        )
