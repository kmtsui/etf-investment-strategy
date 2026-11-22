"""Unit and invariant tests for the buy-only annual backtesting engine and metrics."""

import math

import numpy as np
import pandas as pd
import pytest

from etf_strategy.config import ModelingAssumptions
from etf_strategy.engine import (
    assert_prefix_invariance,
    run_backtest,
    verify_backtest_invariants,
)
from etf_strategy.metrics import compute_xirr, summarize_backtest


def _make_price_df(
    start: str = "2020-01-01",
    end: str = "2022-12-31",
    *,
    drift_per_day: float = 0.0004,
) -> pd.DataFrame:
    dates = pd.bdate_range(start=start, end=end)
    n = len(dates)
    t = np.arange(n, dtype=float)
    # Deterministic smooth wave + positive drift for predictable test assertions.
    prices = 100.0 * np.exp(drift_per_day * t + 0.05 * np.sin(t / 15.0))
    return pd.DataFrame(
        {
            "open": prices,
            "high": prices * 1.01,
            "low": prices * 0.99,
            "close": prices,
            "adj_close": prices,
            "volume": 1_000_000.0,
        },
        index=pd.DatetimeIndex(dates, name="date"),
    )


def _lump_sum_day1(
    price_history_up_to_t: pd.DataFrame,
    remaining_budget: float,
    trading_days_left: int,
) -> float:
    del price_history_up_to_t, trading_days_left
    return remaining_budget


def _equal_daily_pacing(
    price_history_up_to_t: pd.DataFrame,
    remaining_budget: float,
    trading_days_left: int,
) -> float:
    del price_history_up_to_t
    return remaining_budget / trading_days_left


def _never_buy(
    price_history_up_to_t: pd.DataFrame,
    remaining_budget: float,
    trading_days_left: int,
) -> float:
    del price_history_up_to_t, remaining_budget, trading_days_left
    return 0.0


def test_lump_sum_exact_accounting_and_invariants() -> None:
    df = _make_price_df("2020-01-01", "2022-12-31")
    assumptions = ModelingAssumptions(annual_budget=12_000.0)
    res = run_backtest(df, _lump_sum_day1, strategy_name="lump_sum", assumptions=assumptions)

    verify_backtest_invariants(res)
    assert list(res.annual.index) == [2020, 2021, 2022]
    assert np.allclose(res.annual["total_spent"], 12_000.0)
    assert np.allclose(res.annual["ending_cash"], 0.0)
    assert np.allclose(res.annual["forced_sweep"], 0.0)
    assert np.allclose(res.annual["weighted_avg_buy_day"], 1.0)

    # Verify hand-calculated share accumulation on first trading day of each year.
    expected_shares = sum(
        12_000.0 / float(df[df.index.year == y]["adj_close"].iloc[0]) for y in (2020, 2021, 2022)
    )
    assert math.isclose(
        float(res.daily["cumulative_shares"].iloc[-1]), expected_shares, rel_tol=0.0
    )


def test_mandatory_year_end_sweep_deploys_100_percent() -> None:
    df = _make_price_df("2021-01-01", "2022-12-31")
    res = run_backtest(df, _never_buy, strategy_name="dec_sweep")

    verify_backtest_invariants(res)
    assert np.allclose(res.annual["forced_sweep"], 12_000.0)
    assert np.allclose(res.annual["forced_sweep_pct"], 1.0)
    assert np.allclose(res.annual["ending_cash"], 0.0)

    # Every share must be purchased at each year's final trading day price.
    for year in (2021, 2022):
        last_price = float(df[df.index.year == year]["adj_close"].iloc[-1])
        expected_year_shares = 12_000.0 / last_price
        assert math.isclose(float(res.annual.loc[year, "shares_bought"]), expected_year_shares)


def test_buy_only_and_budget_violation_guards() -> None:
    df = _make_price_df("2021-01-01", "2021-12-31")

    def negative_spender(hist: pd.DataFrame, rem: float, left: int) -> float:
        del hist, rem, left
        return -10.0

    with pytest.raises(ValueError, match="Buy-only violation"):
        run_backtest(df, negative_spender, strategy_name="bad_neg")

    def overspender(hist: pd.DataFrame, rem: float, left: int) -> float:
        del hist, left
        return rem + 50.0

    with pytest.raises(ValueError, match="Budget violation"):
        run_backtest(df, overspender, strategy_name="bad_over")

    def nan_spender(hist: pd.DataFrame, rem: float, left: int) -> float:
        del hist, rem, left
        return float("nan")

    with pytest.raises(ValueError, match="non-finite"):
        run_backtest(df, nan_spender, strategy_name="bad_nan")


def test_warmup_history_preserved_without_evaluating_partial_year() -> None:
    # 2019 is partial (starts July); 2020-2021 are complete.
    df = _make_price_df("2019-07-01", "2021-12-31")
    observed_first_len: list[int] = []

    def inspect_history(hist: pd.DataFrame, rem: float, left: int) -> float:
        if not observed_first_len:
            observed_first_len.append(len(hist))
        return rem / left

    res = run_backtest(df, inspect_history, strategy_name="warmup_check")
    assert list(res.annual.index) == [2020, 2021]
    # On Day 1 of 2020, strategy sees all 2019 warm-up rows plus Jan 1, 2020.
    n_2019 = int((df.index.year == 2019).sum())
    assert observed_first_len[0] == n_2019 + 1


def test_prefix_invariance_passes_for_causal_strategies() -> None:
    df = _make_price_df("2020-01-01", "2022-12-31")

    def dip_from_ytd_high(hist: pd.DataFrame, rem: float, left: int) -> float:
        cur_year = hist.index[-1].year
        ytd = hist[hist.index.year == cur_year]["adj_close"]
        drawdown = float(ytd.iloc[-1] / ytd.max()) - 1.0
        if drawdown <= -0.05:
            return rem * 0.25
        return 0.0 if left > 1 else rem

    assert_prefix_invariance(_lump_sum_day1, df)
    assert_prefix_invariance(_equal_daily_pacing, df)
    assert_prefix_invariance(dip_from_ytd_high, df)


def test_prefix_invariance_catches_stateful_and_lookahead_bugs() -> None:
    df = _make_price_df("2020-01-01", "2021-12-31")

    # Impure stateful strategy: calling it again on a truncated prefix changes output.
    calls = {"count": 0}

    def stateful_leaky(hist: pd.DataFrame, rem: float, left: int) -> float:
        del hist, left
        calls["count"] += 1
        return rem * (0.1 if calls["count"] % 2 == 1 else 0.2)

    with pytest.raises(AssertionError, match="Prefix-invariance violation"):
        assert_prefix_invariance(stateful_leaky, df)

    # Whole-share execution edge case: leaves residual cash < 1 share, still buy-only.
    whole_assumptions = ModelingAssumptions(annual_budget=12_000.0, fractional_shares=False)
    res_whole = run_backtest(
        df,
        _lump_sum_day1,
        strategy_name="whole_lump",
        assumptions=whole_assumptions,
    )
    verify_backtest_invariants(res_whole)
    assert np.all(res_whole.daily["shares_bought"] == np.floor(res_whole.daily["shares_bought"]))


def test_cash_yield_accrues_without_lookahead_and_deploys_fully() -> None:
    df = _make_price_df("2021-01-01", "2021-12-31", drift_per_day=0.0)
    # 5% annualized T-bill rate.
    rf = pd.Series(5.0, index=df.index, name="close")

    res_lump = run_backtest(df, _lump_sum_day1, strategy_name="lump", cash_yield_series=rf)
    res_wait = run_backtest(df, _never_buy, strategy_name="wait", cash_yield_series=rf)

    verify_backtest_invariants(res_lump)
    verify_backtest_invariants(res_wait)

    # Day-1 Lump Sum holds zero intra-year cash, so it earns 0 interest.
    assert math.isclose(float(res_lump.annual["interest_earned"].iloc[0]), 0.0, abs_tol=1e-9)

    # Holding $12,000 all year at ~5% simple daily accrual earns roughly $600 (~$590-$615).
    wait_interest = float(res_wait.annual["interest_earned"].iloc[0])
    assert 580.0 < wait_interest < 625.0
    assert math.isclose(
        float(res_wait.annual["total_spent"].iloc[0]),
        12_000.0 + wait_interest,
        abs_tol=1e-6,
    )

    # Verify zero look-ahead on yield: spiking yield on the LAST day must not affect prior days.
    rf_spiked = rf.copy()
    rf_spiked.iloc[-1] = 25.0
    res_spiked = run_backtest(
        df, _never_buy, strategy_name="wait_spike", cash_yield_series=rf_spiked
    )
    pd.testing.assert_series_equal(
        res_wait.daily["interest_earned"],
        res_spiked.daily["interest_earned"],
    )


def test_xirr_and_summary_metrics() -> None:
    # Exact 10% annualized return over 365 days.
    cfs = [
        (pd.Timestamp("2020-01-01"), -1000.0),
        (pd.Timestamp("2020-12-31"), 1100.0),
    ]
    assert math.isclose(compute_xirr(cfs), 0.10, abs_tol=1e-5)

    # Positive steady exponential growth: Day-1 Lump Sum must beat daily pacing and Dec sweep.
    df = _make_price_df("2020-01-01", "2022-12-31", drift_per_day=0.0006)
    m_lump = summarize_backtest(run_backtest(df, _lump_sum_day1, strategy_name="lump"))
    m_daily = summarize_backtest(run_backtest(df, _equal_daily_pacing, strategy_name="daily"))
    m_wait = summarize_backtest(run_backtest(df, _never_buy, strategy_name="wait"))

    assert m_lump["terminal_value"] > m_daily["terminal_value"] > m_wait["terminal_value"]
    assert m_lump["xirr"] > m_daily["xirr"] > m_wait["xirr"]
    assert (
        m_lump["average_cost_basis"] < m_daily["average_cost_basis"] < m_wait["average_cost_basis"]
    )
    assert m_lump["twr_max_drawdown"] <= 0.0
    assert m_lump["twr_annualized_vol"] > 0.0
