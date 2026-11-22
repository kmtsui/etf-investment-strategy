"""Automated invariant, prefix-invariance, oracle-bound, and behavior tests for all strategies."""

import numpy as np
import pandas as pd
import pytest

from etf_strategy.engine import assert_prefix_invariance, run_backtest, verify_backtest_invariants
from etf_strategy.metrics import summarize_backtest
from etf_strategy.strategies import (
    build_default_strategy_registry,
    make_monthly_dca,
    make_oracle_strategy,
    make_quarterly_dca,
)
from etf_strategy.synthetic import generate_gbm_paths, generate_stress_archetype


def _make_multi_regime_df() -> pd.DataFrame:
    """Create a 4-year synthetic dataset with dips, rallies, and warm-up history."""
    paths = generate_gbm_paths(n_years=4, n_paths=1, annual_drift=0.06, annual_vol=0.22, seed=77)
    return paths[0]


@pytest.mark.parametrize(
    "spec",
    build_default_strategy_registry(),
    ids=lambda s: s.name,
)
def test_every_registered_causal_strategy_passes_invariants_and_prefix_invariance(spec) -> None:
    df = _make_multi_regime_df()
    assert spec.is_causal is True
    res = run_backtest(df, spec.fn, strategy_name=spec.name)
    verify_backtest_invariants(res)
    assert_prefix_invariance(spec.fn, df)


def test_oracle_bounds_strictly_sandwich_every_causal_strategy() -> None:
    df = generate_stress_archetype("lost_decade", seed=31, n_years=4)
    oracle_best = make_oracle_strategy(df, mode="best")
    oracle_worst = make_oracle_strategy(df, mode="worst")

    assert oracle_best.is_causal is False
    assert oracle_worst.is_causal is False

    res_best = run_backtest(df, oracle_best.fn, strategy_name=oracle_best.name, is_causal=False)
    res_worst = run_backtest(df, oracle_worst.fn, strategy_name=oracle_worst.name, is_causal=False)
    verify_backtest_invariants(res_best)
    verify_backtest_invariants(res_worst)

    best_shares = res_best.annual["shares_bought"]
    worst_shares = res_worst.annual["shares_bought"]
    assert (best_shares > worst_shares).all()

    for spec in build_default_strategy_registry():
        res = run_backtest(df, spec.fn, strategy_name=spec.name)
        shares = res.annual["shares_bought"]
        assert (shares <= best_shares + 1e-8).all(), f"{spec.name} exceeded Oracle Best Day!"
        assert (shares >= worst_shares - 1e-8).all(), f"{spec.name} fell below Oracle Worst Day!"


def test_monthly_and_quarterly_dca_exact_trade_counts() -> None:
    df = _make_multi_regime_df()

    res_m1 = run_backtest(df, make_monthly_dca(1), strategy_name="m1")
    res_m15 = run_backtest(df, make_monthly_dca(15), strategy_name="m15")
    res_q = run_backtest(df, make_quarterly_dca(), strategy_name="q")

    for year in res_m1.annual.index:
        m1_buys = int((res_m1.daily[res_m1.daily["year"] == year]["spend"] > 0).sum())
        m15_buys = int((res_m15.daily[res_m15.daily["year"] == year]["spend"] > 0).sum())
        q_buys = int((res_q.daily[res_q.daily["year"] == year]["spend"] > 0).sum())
        assert m1_buys == 12
        assert m15_buys == 12
        assert q_buys == 4

        # Each monthly installment must equal $1,000 under $12,000 budget & 0% cash yield.
        m1_spends = res_m1.daily[(res_m1.daily["year"] == year) & (res_m1.daily["spend"] > 0)][
            "spend"
        ]
        np.testing.assert_allclose(m1_spends.to_numpy(), 1_000.0, atol=1e-6)


def test_gbm_null_hypothesis_across_all_causal_strategies() -> None:
    """No causal strategy in the registry may systematically beat Day-1 Lump Sum on positive GBM."""
    paths = generate_gbm_paths(
        n_years=3,
        n_paths=40,
        annual_drift=0.10,
        annual_vol=0.16,
        seed=404,
    )
    registry = build_default_strategy_registry()

    means: dict[str, float] = {}
    for spec in registry:
        vals = [summarize_backtest(run_backtest(df, spec.fn))["terminal_value"] for df in paths]
        means[spec.name] = float(np.mean(vals))

    lump_mean = means["lump_sum_day1"]
    for name, mean_val in means.items():
        if name == "lump_sum_day1":
            continue
        assert mean_val <= lump_mean + 1e-6, (
            f"GBM null violation: '{name}' mean={mean_val:.2f} > lump_sum={lump_mean:.2f}"
        )
