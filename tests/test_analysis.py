"""Verification tests for the analysis pipeline and vectorized Monte Carlo equivalence."""

import numpy as np
import pandas as pd

from etf_strategy.analysis import (
    compute_regime_breakdown,
    compute_rolling_window_summary,
    evaluate_vectorized_mc_strategies,
    run_strategy_suite,
)
from etf_strategy.engine import run_backtest
from etf_strategy.metrics import summarize_backtest
from etf_strategy.strategies import build_default_strategy_registry
from etf_strategy.synthetic import generate_gbm_paths


def test_vectorized_mc_matches_exact_engine_to_machine_precision() -> None:
    """Every vectorized MC strategy must match `run_backtest` on terminal portfolio value."""
    paths = generate_gbm_paths(
        n_years=3,
        n_paths=3,
        annual_drift=0.06,
        annual_vol=0.22,
        seed=909,
    )
    dates = paths[0].index
    price_matrix = np.vstack([p["adj_close"].to_numpy(dtype=float) for p in paths])

    vec_out = evaluate_vectorized_mc_strategies(dates, price_matrix, annual_budget=12_000.0)
    registry = {spec.name: spec for spec in build_default_strategy_registry(annual_budget=12_000.0)}

    for name, vec_vals in vec_out.items():
        spec = registry[name]
        exact_vals = np.array(
            [
                summarize_backtest(run_backtest(df, spec.fn, strategy_name=name))["terminal_value"]
                for df in paths
            ],
            dtype=float,
        )
        np.testing.assert_allclose(
            vec_vals,
            exact_vals,
            rtol=1e-9,
            atol=1e-6,
            err_msg=f"Vectorized MC mismatch for strategy '{name}'",
        )


def test_rolling_window_and_regime_helpers_on_synthetic_data() -> None:
    df = generate_gbm_paths(n_years=6, n_paths=1, annual_drift=0.07, annual_vol=0.20, seed=515)[0]
    table, results = run_strategy_suite(df, include_oracles=True)

    assert "lump_sum_day1" in table.index
    assert "monthly_dca_day1" in table.index
    assert "oracle_best_day" in table.index

    rolling_3y = compute_rolling_window_summary(results, window_years=3)
    assert not rolling_3y.empty
    assert (rolling_3y["n_windows"] == 4).all()
    assert ((rolling_3y["win_rate_vs_lump"] >= 0.0) & (rolling_3y["win_rate_vs_lump"] <= 1.0)).all()

    regimes = compute_regime_breakdown(df, results)
    assert not regimes.empty
    assert set(regimes["regime"].unique()).issubset(
        {"strong_bull (>=15%)", "moderate_bull (0-15%)", "bear (<0%)"}
    )
    assert isinstance(df, pd.DataFrame)
