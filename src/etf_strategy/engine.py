"""Buy-only annual ETF backtesting engine with strict invariant enforcement."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from etf_strategy.config import DEFAULT_ASSUMPTIONS, ModelingAssumptions
from etf_strategy.data import complete_calendar_years
from etf_strategy.types import StrategyFn


@dataclass(frozen=True)
class BacktestResult:
    """Structured output of a multi-year buy-only annual backtest."""

    strategy_name: str
    is_causal: bool
    assumptions: ModelingAssumptions
    daily: pd.DataFrame
    annual: pd.DataFrame


def _align_risk_free_yield(
    dates: pd.DatetimeIndex,
    cash_yield_series: pd.Series | None,
    constant_yield_pct: float,
) -> np.ndarray:
    """Return prior-day annualized yield (%) aligned to trading dates with zero look-ahead."""
    if cash_yield_series is None:
        return np.full(len(dates), float(constant_yield_pct), dtype=float)

    rf = cash_yield_series.copy()
    rf.index = pd.DatetimeIndex(rf.index).normalize()
    rf = rf.sort_index()
    # Reindex to union, forward-fill known past rates only, then shift(1) so day t uses
    # strictly the yield observable by the close of day t-1.
    union_idx = rf.index.union(dates)
    aligned = rf.reindex(union_idx).ffill().shift(1).reindex(dates).fillna(0.0)
    return aligned.to_numpy(dtype=float)


def run_backtest(
    price_df: pd.DataFrame,
    strategy: StrategyFn,
    *,
    strategy_name: str = "unnamed",
    assumptions: ModelingAssumptions = DEFAULT_ASSUMPTIONS,
    cash_yield_series: pd.Series | None = None,
    start_year: int | None = None,
    end_year: int | None = None,
    is_causal: bool = True,
    verify_invariants: bool = True,
) -> BacktestResult:
    """Execute a buy-only fixed-annual-budget backtest over complete calendar years.

    Args:
        price_df: Historical market DataFrame sorted by DatetimeIndex. May include
            pre-start warm-up rows so causal indicators (e.g. 200-day SMA) are populated
            on Day 1 of the first evaluated year.
        strategy: Pure callable `(price_history_up_to_t, remaining_budget, trading_days_left)
            -> spend_t`.
        strategy_name: Identifier for reporting.
        assumptions: ModelingAssumptions controlling budget, price column, shares, and fees.
        cash_yield_series: Optional annualized risk-free yield series in percent (e.g. `^IRX`
            close column). When None, `assumptions.baseline_cash_yield` is used.
        start_year: Optional inclusive first calendar year to evaluate.
        end_year: Optional inclusive last calendar year to evaluate.
        is_causal: Must be True for tradable strategies; False only for hindsight references.
        verify_invariants: When True, runs `verify_backtest_invariants` before returning.
    """
    if price_df.empty:
        raise ValueError("price_df must not be empty.")
    if not isinstance(price_df.index, pd.DatetimeIndex):
        raise ValueError("price_df index must be a DatetimeIndex.")
    if not price_df.index.is_monotonic_increasing or price_df.index.has_duplicates:
        raise ValueError("price_df index must be strictly increasing with unique dates.")

    price_col = assumptions.execution_price_col
    if price_col not in price_df.columns:
        raise ValueError(f"Execution price column '{price_col}' not found in price_df.")

    if assumptions.complete_years_only:
        eval_years = complete_calendar_years(price_df)
    else:
        eval_years = sorted({int(y) for y in price_df.index.year})

    if start_year is not None:
        eval_years = [y for y in eval_years if y >= start_year]
    if end_year is not None:
        eval_years = [y for y in eval_years if y <= end_year]

    if not eval_years:
        raise ValueError("No eligible calendar years found in price_df for the requested range.")

    eval_mask = price_df.index.year.isin(eval_years)
    eval_positions = np.flatnonzero(eval_mask)
    eval_dates = price_df.index[eval_positions]
    n_days = len(eval_positions)

    prices_all = price_df[price_col].to_numpy(dtype=float)
    rf_annual_pct = _align_risk_free_yield(
        eval_dates,
        cash_yield_series,
        assumptions.baseline_cash_yield,
    )

    # Pre-allocate daily output arrays for speed across Monte Carlo runs.
    years_arr = eval_dates.year.to_numpy(dtype=int)
    price_arr = prices_all[eval_positions]
    budget_added_arr = np.zeros(n_days, dtype=float)
    interest_arr = np.zeros(n_days, dtype=float)
    requested_spend_arr = np.zeros(n_days, dtype=float)
    spend_arr = np.zeros(n_days, dtype=float)
    forced_sweep_arr = np.zeros(n_days, dtype=float)
    shares_bought_arr = np.zeros(n_days, dtype=float)
    cum_shares_arr = np.zeros(n_days, dtype=float)
    cash_balance_arr = np.zeros(n_days, dtype=float)
    equity_value_arr = np.zeros(n_days, dtype=float)
    total_value_arr = np.zeros(n_days, dtype=float)
    trading_days_left_arr = np.zeros(n_days, dtype=int)

    # Map each calendar year to its slice inside the eval arrays.
    year_slices: dict[int, tuple[int, int]] = {}
    cursor = 0
    for year in eval_years:
        start_i = cursor
        while cursor < n_days and years_arr[cursor] == year:
            cursor += 1
        year_slices[year] = (start_i, cursor)

    annual_records: list[dict[str, float | int]] = []
    cum_shares = 0.0
    tol = assumptions.budget_tolerance
    commission = assumptions.commission_per_trade
    fractional = assumptions.fractional_shares
    annual_budget = assumptions.annual_budget

    for year in eval_years:
        start_i, end_i = year_slices[year]
        year_len = end_i - start_i
        remaining_budget = annual_budget
        year_interest = 0.0
        year_spent = 0.0
        year_shares_bought = 0.0
        year_forced_sweep = 0.0
        weighted_day_num = 0.0

        for offset in range(year_len):
            i = start_i + offset
            pos = int(eval_positions[i])
            days_left = year_len - offset
            trading_days_left_arr[i] = days_left
            price_t = price_arr[i]

            if not math.isfinite(price_t) or price_t <= 0.0:
                raise ValueError(f"Invalid execution price {price_t} on {eval_dates[i]}.")

            if offset == 0:
                budget_added_arr[i] = annual_budget
            else:
                cal_days = (eval_dates[i] - eval_dates[i - 1]).days
                rate_pct = rf_annual_pct[i]
                if rate_pct != 0.0 and remaining_budget > 0.0 and cal_days > 0:
                    interest_t = remaining_budget * (rate_pct / 100.0) * (cal_days / 365.0)
                    remaining_budget += interest_t
                    interest_arr[i] = interest_t
                    year_interest += interest_t

            # Strict prefix view: strategy can only observe rows <= pos.
            history_up_to_t = price_df.iloc[: pos + 1]
            raw_spend = float(strategy(history_up_to_t, remaining_budget, days_left))

            if not math.isfinite(raw_spend):
                raise ValueError(
                    f"Strategy '{strategy_name}' returned non-finite spend {raw_spend} "
                    f"on {eval_dates[i]}."
                )
            if raw_spend < -1e-9:
                raise ValueError(
                    f"Buy-only violation: strategy '{strategy_name}' returned negative spend "
                    f"{raw_spend} on {eval_dates[i]}."
                )
            if raw_spend > remaining_budget + 1e-9:
                raise ValueError(
                    f"Budget violation: strategy '{strategy_name}' requested {raw_spend:.6f} "
                    f"with only {remaining_budget:.6f} remaining on {eval_dates[i]}."
                )

            requested_spend = min(max(raw_spend, 0.0), remaining_budget)
            requested_spend_arr[i] = requested_spend

            # Mandatory year-end sweep on the final trading day of December.
            if days_left == 1 and remaining_budget > requested_spend:
                forced_sweep = remaining_budget - requested_spend
                spend_t = remaining_budget
            else:
                forced_sweep = 0.0
                spend_t = requested_spend

            if spend_t > 0.0:
                if fractional:
                    net_cash = max(spend_t - commission, 0.0)
                    shares_t = net_cash / price_t
                    actual_spend = spend_t
                else:
                    net_cash = max(spend_t - commission, 0.0)
                    shares_t = float(math.floor(net_cash / price_t))
                    actual_spend = (shares_t * price_t + commission) if shares_t > 0.0 else 0.0
            else:
                shares_t = 0.0
                actual_spend = 0.0

            remaining_budget = max(remaining_budget - actual_spend, 0.0)
            if remaining_budget < tol and days_left == 1 and fractional:
                remaining_budget = 0.0

            cum_shares += shares_t
            year_spent += actual_spend
            year_shares_bought += shares_t
            year_forced_sweep += forced_sweep
            weighted_day_num += actual_spend * (offset + 1)

            spend_arr[i] = actual_spend
            forced_sweep_arr[i] = forced_sweep
            shares_bought_arr[i] = shares_t
            cum_shares_arr[i] = cum_shares
            cash_balance_arr[i] = remaining_budget
            equity_value_arr[i] = cum_shares * price_t
            total_value_arr[i] = equity_value_arr[i] + remaining_budget

        total_available = annual_budget + year_interest
        annual_records.append(
            {
                "year": year,
                "trading_days": year_len,
                "annual_budget": annual_budget,
                "interest_earned": year_interest,
                "total_spent": year_spent,
                "ending_cash": remaining_budget,
                "shares_bought": year_shares_bought,
                "year_avg_cost": (
                    year_spent / year_shares_bought if year_shares_bought > 0 else np.nan
                ),
                "forced_sweep": year_forced_sweep,
                "forced_sweep_pct": (
                    year_forced_sweep / total_available if total_available > 0 else 0.0
                ),
                "weighted_avg_buy_day": (
                    weighted_day_num / year_spent if year_spent > 0 else np.nan
                ),
                "year_end_price": price_arr[end_i - 1],
                "year_end_shares": cum_shares,
                "year_end_equity": equity_value_arr[end_i - 1],
            }
        )

    daily = pd.DataFrame(
        {
            "year": years_arr,
            "trading_days_left": trading_days_left_arr,
            "price": price_arr,
            "budget_added": budget_added_arr,
            "interest_earned": interest_arr,
            "requested_spend": requested_spend_arr,
            "spend": spend_arr,
            "forced_sweep": forced_sweep_arr,
            "shares_bought": shares_bought_arr,
            "cumulative_shares": cum_shares_arr,
            "cash_balance": cash_balance_arr,
            "equity_value": equity_value_arr,
            "total_value": total_value_arr,
        },
        index=eval_dates,
    )
    daily.index.name = "date"

    annual = pd.DataFrame(annual_records).set_index("year")
    result = BacktestResult(
        strategy_name=strategy_name,
        is_causal=is_causal,
        assumptions=assumptions,
        daily=daily,
        annual=annual,
    )

    if verify_invariants:
        verify_backtest_invariants(result, tolerance=tol)

    return result


def verify_backtest_invariants(result: BacktestResult, *, tolerance: float = 1e-6) -> None:
    """Assert non-negotiable buy-only and annual budget-exhaustion invariants."""
    daily = result.daily
    annual = result.annual

    if daily.empty or annual.empty:
        raise AssertionError("Backtest result contains empty daily or annual ledgers.")

    # Buy-only invariants.
    if (daily["spend"] < -tolerance).any():
        raise AssertionError("Negative daily spend detected (buy-only violation).")
    if (daily["shares_bought"] < -tolerance).any():
        raise AssertionError("Negative daily shares bought detected (buy-only violation).")
    if (daily["cash_balance"] < -tolerance).any():
        raise AssertionError("Negative cash balance detected.")
    if (daily["cumulative_shares"].diff().dropna() < -tolerance).any():
        raise AssertionError("Cumulative shares decreased (selling violation).")

    # Exact annual budget exhaustion (strict under fractional shares).
    if result.assumptions.fractional_shares:
        expected_spend = annual["annual_budget"] + annual["interest_earned"]
        spend_err = (annual["total_spent"] - expected_spend).abs()
        if (spend_err > tolerance).any():
            bad_years = spend_err[spend_err > tolerance].to_dict()
            raise AssertionError(f"Annual budget exhaustion failed for years: {bad_years}")
        if (annual["ending_cash"].abs() > tolerance).any():
            bad_cash = annual["ending_cash"][annual["ending_cash"].abs() > tolerance].to_dict()
            raise AssertionError(f"Non-zero ending cash left at year-end: {bad_cash}")


def assert_prefix_invariance(
    strategy: StrategyFn,
    price_df: pd.DataFrame,
    *,
    assumptions: ModelingAssumptions = DEFAULT_ASSUMPTIONS,
    sample_indices: Sequence[int] | None = None,
    atol: float = 1e-12,
) -> None:
    """Mechanically verify that a strategy has zero look-ahead bias and no hidden state leak.

    Performs two checks:
    1. Pointwise truncation check: calling `strategy` on a freshly copied prefix ending at
       day t yields the exact same `requested_spend` as during the full-history backtest.
    2. Prefix-year backtest check: running `run_backtest` on data truncated after the midpoint
       calendar year yields an identical daily ledger through that year.
    """
    full_result = run_backtest(
        price_df,
        strategy,
        strategy_name="prefix_full",
        assumptions=assumptions,
    )
    daily = full_result.daily
    n_days = len(daily)

    if sample_indices is None:
        step = max(1, n_days // 25)
        indices = sorted(set(range(0, n_days, step)) | {0, n_days - 1})
    else:
        indices = list(sample_indices)

    date_to_pos = {dt: pos for pos, dt in enumerate(price_df.index)}

    for idx in indices:
        row = daily.iloc[idx]
        dt = daily.index[idx]
        pos = date_to_pos[dt]
        # Reconstruct exact remaining_budget immediately before strategy execution on day t.
        remaining_before_t = float(row["cash_balance"] + row["spend"])
        days_left = int(row["trading_days_left"])

        truncated_df = price_df.iloc[: pos + 1].copy()
        truncated_spend = float(strategy(truncated_df, remaining_before_t, days_left))
        full_spend = float(row["requested_spend"])

        if not math.isclose(truncated_spend, full_spend, rel_tol=0.0, abs_tol=atol):
            raise AssertionError(
                f"Prefix-invariance violation on {dt.strftime('%Y-%m-%d')}: "
                f"full-history spend={full_spend:.10f}, truncated spend={truncated_spend:.10f}."
            )

    years = list(full_result.annual.index)
    if len(years) >= 2:
        mid_year = years[len(years) // 2 - 1]
        truncated_prices = price_df[price_df.index.year <= mid_year].copy()
        prefix_result = run_backtest(
            truncated_prices,
            strategy,
            strategy_name="prefix_half",
            assumptions=assumptions,
        )
        full_prefix_daily = full_result.daily[full_result.daily["year"] <= mid_year]
        pd.testing.assert_frame_equal(prefix_result.daily, full_prefix_daily, atol=atol, rtol=0.0)
