"""Benchmarks, theoretical oracle bounds, and causal buy-only investment strategies.

Every causal strategy conforms to the pure signature mandated by AGENTS.md:
    (price_history_up_to_t, remaining_budget, trading_days_left) -> spend_t
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from etf_strategy.config import DEFAULT_ASSUMPTIONS
from etf_strategy.types import StrategyFn


@dataclass(frozen=True)
class StrategySpec:
    """Metadata wrapper pairing a strategy callable with its category and causality flag."""

    name: str
    category: str
    fn: StrategyFn
    is_causal: bool = True
    description: str = ""


def _is_first_trading_day_of_year(hist: pd.DataFrame) -> bool:
    if len(hist) <= 1:
        return True
    return bool(hist.index[-1].year != hist.index[-2].year)


def _ytd_trading_day_index(hist: pd.DataFrame) -> int:
    """Return 0-based trading day offset within the current calendar year."""
    cur_year = hist.index[-1].year
    years = hist.index.year
    # Scan backwards from the end (fast O(ytd_days), max ~260 steps).
    idx = len(hist) - 1
    while idx > 0 and years[idx - 1] == cur_year:
        idx -= 1
    return (len(hist) - 1) - idx


def _trailing_drawdown(hist: pd.DataFrame, *, window: int | None, price_col: str) -> float:
    """Return drawdown (<= 0.0) of latest price vs trailing high (or YTD high if window is None)."""
    prices = hist[price_col].to_numpy(dtype=float)
    if window is None:
        ytd_len = _ytd_trading_day_index(hist) + 1
        ref = prices[-ytd_len:]
    else:
        ref = prices[-min(len(prices), window) :]
    peak = float(np.max(ref))
    if peak <= 0.0:
        return 0.0
    return float(prices[-1] / peak) - 1.0


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------


def lump_sum_day1(
    price_history_up_to_t: pd.DataFrame,
    remaining_budget: float,
    trading_days_left: int,
) -> float:
    """Benchmark 1: Invest 100% of the annual budget on the first trading day of the year."""
    del price_history_up_to_t, trading_days_left
    return remaining_budget


def make_monthly_dca(target_day: int = 1) -> StrategyFn:
    """Benchmark 2 factory: Invest 1/12 of annual budget on `target_day` each month.

    If the market is closed on `target_day`, execution rolls forward to the next trading day.
    `target_day` must be in [1, 28] so every calendar month has a valid trigger date.
    """
    if not 1 <= target_day <= 28:
        raise ValueError("target_day must be between 1 and 28 inclusive.")

    def _strategy(
        price_history_up_to_t: pd.DataFrame,
        remaining_budget: float,
        trading_days_left: int,
    ) -> float:
        del trading_days_left
        if remaining_budget <= 0.0:
            return 0.0

        dt = price_history_up_to_t.index[-1]
        if dt.day < target_day:
            return 0.0

        if len(price_history_up_to_t) == 1:
            should_buy = True
        else:
            prev_dt = price_history_up_to_t.index[-2]
            new_month = (prev_dt.year != dt.year) or (prev_dt.month != dt.month)
            crossed_target = (prev_dt.month == dt.month) and (prev_dt.day < target_day)
            should_buy = new_month or crossed_target

        if not should_buy:
            return 0.0

        months_left_incl = 13 - dt.month
        return remaining_budget / months_left_incl

    return _strategy


# ---------------------------------------------------------------------------
# Theoretical Hindsight Bounds (Reference Only, is_causal = False)
# ---------------------------------------------------------------------------


def make_oracle_strategy(
    full_price_df: pd.DataFrame,
    *,
    mode: str = "best",
    price_col: str = DEFAULT_ASSUMPTIONS.execution_price_col,
) -> StrategySpec:
    """Create a non-causal hindsight reference strategy ('best' or 'worst' day of each year)."""
    if mode not in ("best", "worst"):
        raise ValueError("mode must be 'best' or 'worst'.")

    target_dates: set[pd.Timestamp] = set()
    for _, group in full_price_df.groupby(full_price_df.index.year):
        if group.empty:
            continue
        chosen = group[price_col].idxmin() if mode == "best" else group[price_col].idxmax()
        target_dates.add(pd.Timestamp(chosen))

    def _oracle(
        price_history_up_to_t: pd.DataFrame,
        remaining_budget: float,
        trading_days_left: int,
    ) -> float:
        del trading_days_left
        dt = pd.Timestamp(price_history_up_to_t.index[-1])
        return remaining_budget if dt in target_dates else 0.0

    return StrategySpec(
        name=f"oracle_{mode}_day",
        category="oracle",
        fn=_oracle,
        is_causal=False,
        description=f"Non-causal hindsight bound: 100% invested on each year's {mode} price day.",
    )


# ---------------------------------------------------------------------------
# Calendar & Frequency Variants
# ---------------------------------------------------------------------------


def make_periodic_trading_day_dca(interval_days: int) -> StrategyFn:
    """Invest evenly every `interval_days` trading days (e.g. 5=weekly, 10=biweekly)."""
    if interval_days < 1:
        raise ValueError("interval_days must be >= 1.")

    def _strategy(
        price_history_up_to_t: pd.DataFrame,
        remaining_budget: float,
        trading_days_left: int,
    ) -> float:
        if remaining_budget <= 0.0:
            return 0.0
        day_idx = _ytd_trading_day_index(price_history_up_to_t)
        if day_idx % interval_days != 0:
            return 0.0
        buys_left_incl = math.ceil(trading_days_left / interval_days)
        return remaining_budget / max(buys_left_incl, 1)

    return _strategy


def make_quarterly_dca() -> StrategyFn:
    """Invest 1/4 of the annual budget on the first trading day of Q1, Q2, Q3, and Q4."""
    quarter_months = {1: 4, 4: 3, 7: 2, 10: 1}

    def _strategy(
        price_history_up_to_t: pd.DataFrame,
        remaining_budget: float,
        trading_days_left: int,
    ) -> float:
        del trading_days_left
        if remaining_budget <= 0.0:
            return 0.0
        dt = price_history_up_to_t.index[-1]
        if dt.month not in quarter_months:
            return 0.0
        if len(price_history_up_to_t) > 1:
            prev_dt = price_history_up_to_t.index[-2]
            if prev_dt.year == dt.year and prev_dt.month == dt.month:
                return 0.0
        return remaining_budget / quarter_months[dt.month]

    return _strategy


def make_front_loaded_q1_dca() -> StrategyFn:
    """Deploy the entire annual budget evenly across the first trading day of Jan, Feb, and Mar."""
    q1_months_left = {1: 3, 2: 2, 3: 1}

    def _strategy(
        price_history_up_to_t: pd.DataFrame,
        remaining_budget: float,
        trading_days_left: int,
    ) -> float:
        del trading_days_left
        if remaining_budget <= 0.0:
            return 0.0
        dt = price_history_up_to_t.index[-1]
        if dt.month not in q1_months_left:
            return 0.0
        if len(price_history_up_to_t) > 1:
            prev_dt = price_history_up_to_t.index[-2]
            if prev_dt.year == dt.year and prev_dt.month == dt.month:
                return 0.0
        return remaining_budget / q1_months_left[dt.month]

    return _strategy


# ---------------------------------------------------------------------------
# Drawdown / Dip-Buying Strategies
# ---------------------------------------------------------------------------


def make_drawdown_buyer(
    threshold: float,
    *,
    tranche_frac: float = 1.0,
    window: int | None = None,
    price_col: str = DEFAULT_ASSUMPTIONS.execution_price_col,
) -> StrategyFn:
    """Deploy `tranche_frac * remaining_budget` when drawdown from peak <= -threshold.

    Args:
        threshold: Positive drawdown trigger (e.g. 0.05 for -5%).
        tranche_frac: Fraction of remaining budget to deploy on trigger day, in (0, 1].
        window: Trailing lookback window in trading days; None uses causal YTD high.
    """
    if threshold <= 0.0:
        raise ValueError("threshold must be > 0.")
    if not 0.0 < tranche_frac <= 1.0:
        raise ValueError("tranche_frac must be in (0, 1].")

    def _strategy(
        price_history_up_to_t: pd.DataFrame,
        remaining_budget: float,
        trading_days_left: int,
    ) -> float:
        del trading_days_left
        if remaining_budget <= 0.0:
            return 0.0
        dd = _trailing_drawdown(price_history_up_to_t, window=window, price_col=price_col)
        if dd <= -threshold:
            return remaining_budget * tranche_frac
        return 0.0

    return _strategy


def make_tiered_drawdown_buyer(
    *,
    annual_budget: float = DEFAULT_ASSUMPTIONS.annual_budget,
    tiers: tuple[tuple[float, float], ...] = ((0.05, 0.33), (0.10, 0.66), (0.15, 1.0)),
    window: int = 252,
    price_col: str = DEFAULT_ASSUMPTIONS.execution_price_col,
) -> StrategyFn:
    """Deploy cumulative target budget fractions at deeper drawdown tiers from trailing high.

    `tiers` is a tuple of `(drawdown_threshold, cumulative_deployed_fraction)` pairs.
    Stateless & pure: computes `target_remaining = annual_budget * (1 - target_deployed)`
    and spends `max(0.0, remaining_budget - target_remaining)`.
    """
    ordered_tiers = sorted(tiers, key=lambda pair: pair[0], reverse=True)

    def _strategy(
        price_history_up_to_t: pd.DataFrame,
        remaining_budget: float,
        trading_days_left: int,
    ) -> float:
        del trading_days_left
        if remaining_budget <= 0.0:
            return 0.0
        dd = _trailing_drawdown(price_history_up_to_t, window=window, price_col=price_col)
        target_deployed = 0.0
        for thresh, frac in ordered_tiers:
            if dd <= -thresh:
                target_deployed = frac
                break
        if target_deployed <= 0.0:
            return 0.0
        target_remaining = annual_budget * max(0.0, 1.0 - target_deployed)
        return min(remaining_budget, max(0.0, remaining_budget - target_remaining))

    return _strategy


def make_hybrid_lump_plus_dip(
    *,
    lump_frac: float = 0.5,
    dip_threshold: float = 0.10,
    window: int = 252,
    price_col: str = DEFAULT_ASSUMPTIONS.execution_price_col,
) -> StrategyFn:
    """Invest `lump_frac` on Day 1 of each year and deploy the rest on a `-dip_threshold` dip."""
    if not 0.0 < lump_frac < 1.0:
        raise ValueError("lump_frac must be in (0, 1).")

    def _strategy(
        price_history_up_to_t: pd.DataFrame,
        remaining_budget: float,
        trading_days_left: int,
    ) -> float:
        del trading_days_left
        if remaining_budget <= 0.0:
            return 0.0
        if _is_first_trading_day_of_year(price_history_up_to_t):
            return remaining_budget * lump_frac
        dd = _trailing_drawdown(price_history_up_to_t, window=window, price_col=price_col)
        if dd <= -dip_threshold:
            return remaining_budget
        return 0.0

    return _strategy


# ---------------------------------------------------------------------------
# Technical / Mean-Reversion Pacing Strategies
# ---------------------------------------------------------------------------


def make_below_sma_buyer(
    *,
    window: int = 200,
    tranche_frac: float = 0.10,
    price_col: str = DEFAULT_ASSUMPTIONS.execution_price_col,
) -> StrategyFn:
    """Deploy `tranche_frac * remaining_budget` whenever price trades below its N-day SMA."""
    if window < 2:
        raise ValueError("window must be >= 2.")

    def _strategy(
        price_history_up_to_t: pd.DataFrame,
        remaining_budget: float,
        trading_days_left: int,
    ) -> float:
        del trading_days_left
        if remaining_budget <= 0.0:
            return 0.0
        prices = price_history_up_to_t[price_col].to_numpy(dtype=float)
        if len(prices) < window:
            return 0.0
        sma = float(np.mean(prices[-window:]))
        if prices[-1] < sma:
            return remaining_budget * tranche_frac
        return 0.0

    return _strategy


def make_rsi_buyer(
    *,
    window: int = 14,
    threshold: float = 30.0,
    tranche_frac: float = 0.50,
    price_col: str = DEFAULT_ASSUMPTIONS.execution_price_col,
) -> StrategyFn:
    """Deploy `tranche_frac * remaining_budget` when simple trailing RSI(window) <= threshold."""
    if window < 2:
        raise ValueError("window must be >= 2.")

    def _strategy(
        price_history_up_to_t: pd.DataFrame,
        remaining_budget: float,
        trading_days_left: int,
    ) -> float:
        del trading_days_left
        if remaining_budget <= 0.0:
            return 0.0
        prices = price_history_up_to_t[price_col].to_numpy(dtype=float)
        if len(prices) < window + 1:
            return 0.0
        deltas = np.diff(prices[-(window + 1) :])
        avg_gain = float(np.mean(np.maximum(deltas, 0.0)))
        avg_loss = float(np.mean(np.maximum(-deltas, 0.0)))
        if avg_loss == 0.0:
            rsi = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi = 100.0 - (100.0 / (1.0 + rs))
        if rsi <= threshold:
            return remaining_budget * tranche_frac
        return 0.0

    return _strategy


def make_bollinger_lower_buyer(
    *,
    window: int = 20,
    num_std: float = 2.0,
    tranche_frac: float = 0.50,
    price_col: str = DEFAULT_ASSUMPTIONS.execution_price_col,
) -> StrategyFn:
    """Deploy `tranche_frac * remaining_budget` when price hits the lower Bollinger Band."""
    if window < 2:
        raise ValueError("window must be >= 2.")

    def _strategy(
        price_history_up_to_t: pd.DataFrame,
        remaining_budget: float,
        trading_days_left: int,
    ) -> float:
        del trading_days_left
        if remaining_budget <= 0.0:
            return 0.0
        prices = price_history_up_to_t[price_col].to_numpy(dtype=float)
        if len(prices) < window:
            return 0.0
        seg = prices[-window:]
        mean = float(np.mean(seg))
        std = float(np.std(seg, ddof=1))
        if prices[-1] <= mean - num_std * std:
            return remaining_budget * tranche_frac
        return 0.0

    return _strategy


# ---------------------------------------------------------------------------
# Volatility-Aware & Adaptive Remaining-Budget Pacing
# ---------------------------------------------------------------------------


def make_vol_scaled_pacing(
    *,
    short_window: int = 20,
    long_window: int = 252,
    sensitivity: float = 1.5,
    price_col: str = DEFAULT_ASSUMPTIONS.execution_price_col,
) -> StrategyFn:
    """Scale daily remaining-budget pace by trailing short/long realized volatility ratio."""

    def _strategy(
        price_history_up_to_t: pd.DataFrame,
        remaining_budget: float,
        trading_days_left: int,
    ) -> float:
        if remaining_budget <= 0.0:
            return 0.0
        base_spend = remaining_budget / max(trading_days_left, 1)
        prices = price_history_up_to_t[price_col].to_numpy(dtype=float)
        if len(prices) < short_window + 2:
            return base_spend

        log_ret = np.diff(np.log(prices[-min(len(prices), long_window + 1) :]))
        short_vol = float(np.std(log_ret[-short_window:], ddof=1))
        long_vol = float(np.std(log_ret, ddof=1))
        if long_vol <= 1e-12:
            return base_spend

        ratio = max(short_vol / long_vol, 1e-3)
        mult = min(max(ratio**sensitivity, 0.25), 4.0)
        return min(remaining_budget, base_spend * mult)

    return _strategy


def make_adaptive_drawdown_pacing(
    *,
    window: int = 252,
    boost_per_10pct_dd: float = 4.0,
    price_col: str = DEFAULT_ASSUMPTIONS.execution_price_col,
) -> StrategyFn:
    """Accelerate daily remaining-budget pacing smoothly as trailing drawdown deepens."""

    def _strategy(
        price_history_up_to_t: pd.DataFrame,
        remaining_budget: float,
        trading_days_left: int,
    ) -> float:
        if remaining_budget <= 0.0:
            return 0.0
        base_spend = remaining_budget / max(trading_days_left, 1)
        dd = _trailing_drawdown(price_history_up_to_t, window=window, price_col=price_col)
        mult = 1.0 + boost_per_10pct_dd * (abs(dd) / 0.10)
        return min(remaining_budget, base_spend * mult)

    return _strategy


# ---------------------------------------------------------------------------
# Default Causal Strategy Registry
# ---------------------------------------------------------------------------


def build_default_strategy_registry(
    *,
    annual_budget: float = DEFAULT_ASSUMPTIONS.annual_budget,
    price_col: str = DEFAULT_ASSUMPTIONS.execution_price_col,
) -> list[StrategySpec]:
    """Return the full catalog of causal benchmark and candidate strategies."""
    return [
        # Benchmarks
        StrategySpec(
            name="lump_sum_day1",
            category="benchmark",
            fn=lump_sum_day1,
            description="100% invested on the first trading day of each calendar year.",
        ),
        StrategySpec(
            name="monthly_dca_day1",
            category="benchmark",
            fn=make_monthly_dca(target_day=1),
            description="Equal 1/12 monthly installments on the 1st trading day of each month.",
        ),
        # Calendar & Frequency Variants
        StrategySpec(
            name="monthly_dca_day15",
            category="calendar",
            fn=make_monthly_dca(target_day=15),
            description="Equal 1/12 monthly installments on the 15th (or next trading day).",
        ),
        StrategySpec(
            name="weekly_dca",
            category="calendar",
            fn=make_periodic_trading_day_dca(5),
            description="Equal installments every 5 trading days.",
        ),
        StrategySpec(
            name="biweekly_dca",
            category="calendar",
            fn=make_periodic_trading_day_dca(10),
            description="Equal installments every 10 trading days.",
        ),
        StrategySpec(
            name="quarterly_dca",
            category="calendar",
            fn=make_quarterly_dca(),
            description="Equal 1/4 installments on the first trading day of Q1-Q4.",
        ),
        StrategySpec(
            name="front_loaded_q1",
            category="calendar",
            fn=make_front_loaded_q1_dca(),
            description="Equal 1/3 installments on the first trading day of Jan, Feb, and Mar.",
        ),
        # Drawdown / Dip-Buying
        StrategySpec(
            name="dip_ytd_5pct",
            category="drawdown",
            fn=make_drawdown_buyer(0.05, tranche_frac=1.0, window=None, price_col=price_col),
            description="All-in on first -5% dip from YTD high (else Dec sweep).",
        ),
        StrategySpec(
            name="dip_ytd_10pct",
            category="drawdown",
            fn=make_drawdown_buyer(0.10, tranche_frac=1.0, window=None, price_col=price_col),
            description="All-in on first -10% dip from YTD high (else Dec sweep).",
        ),
        StrategySpec(
            name="dip_tiered_5_10_15",
            category="drawdown",
            fn=make_tiered_drawdown_buyer(annual_budget=annual_budget, price_col=price_col),
            description="Tiered deployment (33%/66%/100%) at -5%/-10%/-15% off 252d high.",
        ),
        StrategySpec(
            name="hybrid_50_lump_dip10",
            category="drawdown",
            fn=make_hybrid_lump_plus_dip(lump_frac=0.5, dip_threshold=0.10, price_col=price_col),
            description="50% invested Day 1; remaining 50% on -10% dip off 252d high (else Dec).",
        ),
        # Technical / Mean-Reversion
        StrategySpec(
            name="below_sma50_all_in",
            category="technical",
            fn=make_below_sma_buyer(window=50, tranche_frac=1.0, price_col=price_col),
            description="All-in on first day price closes below 50-day SMA (else Dec sweep).",
        ),
        StrategySpec(
            name="below_sma200_paced",
            category="technical",
            fn=make_below_sma_buyer(window=200, tranche_frac=0.10, price_col=price_col),
            description="Deploy 10% of remaining budget on each day below 200-day SMA.",
        ),
        StrategySpec(
            name="rsi14_oversold_30",
            category="technical",
            fn=make_rsi_buyer(window=14, threshold=30.0, tranche_frac=0.50, price_col=price_col),
            description="Deploy 50% of remaining budget when 14-day RSI <= 30.",
        ),
        StrategySpec(
            name="bollinger_lower_2sd",
            category="technical",
            fn=make_bollinger_lower_buyer(
                window=20, num_std=2.0, tranche_frac=0.50, price_col=price_col
            ),
            description="Deploy 50% of remaining budget on 20-day lower Bollinger Band (-2 SD).",
        ),
        # Volatility-Aware & Adaptive Pacing
        StrategySpec(
            name="vol_scaled_pacing",
            category="adaptive",
            fn=make_vol_scaled_pacing(price_col=price_col),
            description="Daily remaining-budget pacing scaled by 20d/252d realized vol ratio.",
        ),
        StrategySpec(
            name="adaptive_dd_pacing",
            category="adaptive",
            fn=make_adaptive_drawdown_pacing(price_col=price_col),
            description="Daily remaining-budget pacing accelerated by trailing 252d drawdown.",
        ),
    ]
