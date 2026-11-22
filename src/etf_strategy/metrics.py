"""Portfolio performance, cost-basis, XIRR, and risk metrics for buy-only backtests."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd

from etf_strategy.engine import BacktestResult


def compute_xirr(
    cashflows: Sequence[tuple[pd.Timestamp, float]],
    *,
    tol: float = 1e-9,
    max_iter: int = 200,
) -> float:
    """Compute annualized money-weighted IRR (XIRR) for irregular dated cashflows.

    Cashflows follow investor sign convention: contributions are negative, terminal value
    is positive. Returns NaN if cashflows do not contain both positive and negative values.
    """
    if len(cashflows) < 2:
        return float("nan")

    amounts = np.array([cf[1] for cf in cashflows], dtype=float)
    if not (np.any(amounts < 0.0) and np.any(amounts > 0.0)):
        return float("nan")

    dates = [pd.Timestamp(cf[0]) for cf in cashflows]
    t0 = min(dates)
    years = np.array([(d - t0).days / 365.0 for d in dates], dtype=float)

    def npv(rate: float) -> float:
        return float(np.sum(amounts / np.power(1.0 + rate, years)))

    def d_npv(rate: float) -> float:
        return float(np.sum(-years * amounts / np.power(1.0 + rate, years + 1.0)))

    # Try fast Newton-Raphson first from a sensible 8% initial guess.
    rate = 0.08
    for _ in range(50):
        if rate <= -0.999999:
            break
        val = npv(rate)
        if abs(val) < tol:
            return rate
        deriv = d_npv(rate)
        if abs(deriv) < 1e-14:
            break
        next_rate = rate - val / deriv
        if next_rate <= -0.999999 or not math.isfinite(next_rate):
            break
        if abs(next_rate - rate) < tol:
            return next_rate
        rate = next_rate

    # Guaranteed bracketed bisection fallback.
    low, high = -0.9999, 10.0
    f_low = npv(low)
    f_high = npv(high)
    while f_low * f_high > 0.0 and high < 1e4:
        high *= 2.0
        f_high = npv(high)
    if f_low * f_high > 0.0:
        return float("nan")

    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        f_mid = npv(mid)
        if abs(f_mid) < tol or (high - low) < tol:
            return mid
        if f_low * f_mid <= 0.0:
            high = mid
            f_high = f_mid
        else:
            low = mid
            f_low = f_mid
    return 0.5 * (low + high)


def time_weighted_returns(daily: pd.DataFrame) -> pd.Series:
    """Compute daily time-weighted returns adjusting for external annual budget inflows."""
    prev_total = daily["total_value"].shift(1)
    # Remove fresh Jan Day-1 budget inflow from numerator so external cash does not create a
    # spurious jump in daily returns.
    pre_inflow_value = daily["total_value"] - daily["budget_added"]
    twr = (pre_inflow_value / prev_total) - 1.0
    return twr.iloc[1:].replace([np.inf, -np.inf], np.nan).dropna()


def max_drawdown_from_returns(returns: pd.Series) -> float:
    """Return maximum peak-to-trough drawdown as a negative decimal (e.g. -0.35 for -35%)."""
    if returns.empty:
        return 0.0
    wealth = (1.0 + returns).cumprod()
    peak = wealth.cummax()
    drawdowns = (wealth / peak) - 1.0
    return float(drawdowns.min())


def summarize_backtest(result: BacktestResult) -> dict[str, float]:
    """Compute headline wealth, cost-basis, XIRR, timing, and risk metrics for a backtest.

    Note on XIRR cashflows: each year's fixed annual budget is booked as an outflow on the
    first trading day of that calendar year (when the budget becomes available to invest),
    not on the purchase date. This prevents delaying buys until December from artificially
    inflating XIRR.
    """
    daily = result.daily
    annual = result.annual

    terminal_value = float(daily["total_value"].iloc[-1])
    total_contributed = float(annual["annual_budget"].sum())
    total_interest_earned = float(annual["interest_earned"].sum())
    total_spent = float(annual["total_spent"].sum())
    total_shares = float(daily["cumulative_shares"].iloc[-1])

    # Investor-level cashflows: -annual_budget on Day 1 of each year, +terminal_value at end.
    inflow_rows = daily[daily["budget_added"] > 0.0]
    cashflows: list[tuple[pd.Timestamp, float]] = [
        (pd.Timestamp(dt), -float(amt))
        for dt, amt in zip(inflow_rows.index, inflow_rows["budget_added"], strict=True)
    ]
    cashflows.append((pd.Timestamp(daily.index[-1]), terminal_value))
    xirr = compute_xirr(cashflows)

    twr = time_weighted_returns(daily)
    ann_vol = float(twr.std(ddof=1) * math.sqrt(252.0)) if len(twr) > 1 else 0.0
    mdd = max_drawdown_from_returns(twr)

    return {
        "n_years": float(len(annual)),
        "total_contributed": total_contributed,
        "total_interest_earned": total_interest_earned,
        "total_spent": total_spent,
        "total_shares": total_shares,
        "terminal_value": terminal_value,
        "total_return_multiple": terminal_value / total_contributed,
        "total_return_pct": (terminal_value / total_contributed) - 1.0,
        "average_cost_basis": total_contributed / total_shares,
        "xirr": xirr,
        "twr_annualized_vol": ann_vol,
        "twr_max_drawdown": mdd,
        "avg_forced_sweep_pct": float(annual["forced_sweep_pct"].mean()),
        "avg_weighted_buy_day": float(annual["weighted_avg_buy_day"].mean()),
    }
