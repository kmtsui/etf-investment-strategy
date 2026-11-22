"""Monte Carlo hyperparameter tuning for technical & adaptive strategies, tested on real data.

Follows AGENTS.md Rules 6 & 7:
- Synthetic MC paths (Stationary Block Bootstrap + GJR-GARCH(1,1)-t) are calibrated strictly
  on SPY complete years <= 2015.
- Hyperparameters are selected solely on synthetic MC performance, then evaluated unchanged
  on held-out historical OOS (2016-2025) and full historical SPY/QQQ series.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from etf_strategy.analysis import REPORTS_DIR, _generate_mc_price_matrix
from etf_strategy.config import DEFAULT_ASSUMPTIONS
from etf_strategy.data import filter_complete_years, load_raw_dataset
from etf_strategy.engine import assert_prefix_invariance, run_backtest
from etf_strategy.metrics import summarize_backtest
from etf_strategy.strategies import (
    lump_sum_day1,
    make_adaptive_drawdown_pacing,
    make_below_sma_buyer,
    make_bollinger_lower_buyer,
    make_monthly_dca,
    make_rsi_buyer,
    make_vol_scaled_pacing,
)
from etf_strategy.types import StrategyFn


@dataclass(frozen=True)
class TunedConfig:
    """Best MC-selected hyperparameter configuration for a strategy family."""

    family: str
    params: dict[str, Any]
    mc_mean_vs_lump_pct: float
    mc_mean_vs_dca_pct: float
    mc_win_rate_vs_lump: float
    mc_win_rate_vs_dca: float
    mc_avg_buy_day: float


def _year_slices(dates: pd.DatetimeIndex) -> list[tuple[int, int]]:
    years = dates.year.to_numpy()
    unique_years = sorted({int(y) for y in years})
    slices: list[tuple[int, int]] = []
    for y in unique_years:
        idx = np.flatnonzero(years == y)
        slices.append((int(idx[0]), int(idx[-1]) + 1))
    return slices


def _run_signal_tranche_mc(
    prices: np.ndarray,
    year_slices: list[tuple[int, int]],
    signal_mask: np.ndarray,
    tranche_frac: float,
    annual_budget: float,
) -> tuple[np.ndarray, float]:
    """Vectorized execution for any `if signal: spend tranche_frac * rem` strategy."""
    n_paths = prices.shape[0]
    shares = np.zeros(n_paths, dtype=float)
    weighted_day_sum = 0.0
    total_spend_sum = 0.0

    for start_i, end_i in year_slices:
        year_len = end_i - start_i
        rem = np.full(n_paths, annual_budget, dtype=float)
        for offset in range(year_len):
            t = start_i + offset
            days_left = year_len - offset
            spend = rem if days_left == 1 else np.where(signal_mask[:, t], rem * tranche_frac, 0.0)
            shares += spend / prices[:, t]
            rem = np.maximum(rem - spend, 0.0)
            s_mean = float(np.mean(spend))
            weighted_day_sum += s_mean * (offset + 1)
            total_spend_sum += s_mean

    avg_buy_day = weighted_day_sum / max(total_spend_sum, 1e-12)
    return shares * prices[:, -1], avg_buy_day


def _run_pacing_mult_mc(
    prices: np.ndarray,
    year_slices: list[tuple[int, int]],
    mult_matrix: np.ndarray,
    annual_budget: float,
) -> tuple[np.ndarray, float]:
    """Vectorized execution for any `min(rem, (rem / days_left) * mult_t)` pacing strategy."""
    n_paths = prices.shape[0]
    shares = np.zeros(n_paths, dtype=float)
    weighted_day_sum = 0.0
    total_spend_sum = 0.0

    for start_i, end_i in year_slices:
        year_len = end_i - start_i
        rem = np.full(n_paths, annual_budget, dtype=float)
        for offset in range(year_len):
            t = start_i + offset
            days_left = year_len - offset
            spend = (
                rem if days_left == 1 else np.minimum(rem, (rem / days_left) * mult_matrix[:, t])
            )
            shares += spend / prices[:, t]
            rem = np.maximum(rem - spend, 0.0)
            s_mean = float(np.mean(spend))
            weighted_day_sum += s_mean * (offset + 1)
            total_spend_sum += s_mean

    avg_buy_day = weighted_day_sum / max(total_spend_sum, 1e-12)
    return shares * prices[:, -1], avg_buy_day


def _compute_sma(prices: np.ndarray, window: int) -> np.ndarray:
    sma = np.full_like(prices, np.nan)
    csum = np.cumsum(prices, axis=1)
    sma[:, window - 1] = csum[:, window - 1] / window
    sma[:, window:] = (csum[:, window:] - csum[:, :-window]) / window
    return sma


def _compute_rolling_std(prices: np.ndarray, window: int) -> np.ndarray:
    """Exact sample std (ddof=1) over trailing `window` observations."""
    n_paths, n_days = prices.shape
    out = np.full((n_paths, n_days), np.nan, dtype=float)
    for t in range(window - 1, n_days):
        out[:, t] = np.std(prices[:, t - window + 1 : t + 1], axis=1, ddof=1)
    return out


def _compute_simple_rsi(prices: np.ndarray, window: int) -> np.ndarray:
    n_paths, n_days = prices.shape
    rsi = np.full((n_paths, n_days), np.nan, dtype=float)
    deltas = np.diff(prices, axis=1)
    gains = np.maximum(deltas, 0.0)
    losses = np.maximum(-deltas, 0.0)
    cg = np.cumsum(gains, axis=1)
    cl = np.cumsum(losses, axis=1)
    avg_g = np.empty((n_paths, n_days - window), dtype=float)
    avg_l = np.empty((n_paths, n_days - window), dtype=float)
    avg_g[:, 0] = cg[:, window - 1] / window
    avg_g[:, 1:] = (cg[:, window:] - cg[:, :-window]) / window
    avg_l[:, 0] = cl[:, window - 1] / window
    avg_l[:, 1:] = (cl[:, window:] - cl[:, :-window]) / window
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = np.where(avg_l == 0.0, np.inf, avg_g / avg_l)
        rsi[:, window:] = np.where(avg_l == 0.0, 100.0, 100.0 - (100.0 / (1.0 + rs)))
    return rsi


def _compute_vol_ratio_matrix(
    prices: np.ndarray,
    short_window: int,
    long_window: int,
) -> np.ndarray:
    """Causal trailing short_vol / long_vol matrix matching `make_vol_scaled_pacing`."""
    n_paths, n_days = prices.shape
    log_ret = np.diff(np.log(prices), axis=1)  # shape (n_paths, n_days - 1)
    ratio = np.ones((n_paths, n_days), dtype=float)
    for t in range(short_window + 1, n_days):
        # At day t, prices[:t+1] has t log-returns: log_ret[:, :t].
        lo_long = max(0, t - long_window)
        short_vol = np.std(log_ret[:, t - short_window : t], axis=1, ddof=1)
        long_vol = np.std(log_ret[:, lo_long:t], axis=1, ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(long_vol > 1e-12, np.maximum(short_vol / long_vol, 1e-3), 1.0)
        ratio[:, t] = r
    return ratio


def _compute_trailing_dd_matrix(prices: np.ndarray, window: int) -> np.ndarray:
    n_paths, n_days = prices.shape
    high = np.empty((n_paths, n_days), dtype=float)
    for t in range(n_days):
        lo = max(0, t - window + 1)
        high[:, t] = np.max(prices[:, lo : t + 1], axis=1)
    return (prices / high) - 1.0


def _benchmark_mc_terminals(
    dates: pd.DatetimeIndex,
    prices: np.ndarray,
    annual_budget: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute exact Day-1 Lump Sum and Monthly DCA Day-1 terminal values on price matrix."""
    n_paths, n_days = prices.shape
    slices = _year_slices(dates)
    months = dates.month.to_numpy()
    years = dates.year.to_numpy()
    is_new_month = np.ones(n_days, dtype=bool)
    is_new_month[1:] = (months[1:] != months[:-1]) | (years[1:] != years[:-1])

    lump_shares = np.zeros(n_paths, dtype=float)
    dca_shares = np.zeros(n_paths, dtype=float)

    for start_i, end_i in slices:
        lump_shares += annual_budget / prices[:, start_i]
        rem = np.full(n_paths, annual_budget, dtype=float)
        year_len = end_i - start_i
        for offset in range(year_len):
            t = start_i + offset
            days_left = year_len - offset
            if days_left == 1:
                spend = rem
            elif is_new_month[t]:
                spend = rem / (13 - int(months[t]))
            else:
                spend = np.zeros(n_paths, dtype=float)
            dca_shares += spend / prices[:, t]
            rem = np.maximum(rem - spend, 0.0)

    final_p = prices[:, -1]
    return lump_shares * final_p, dca_shares * final_p


def build_strategy_from_params(family: str, params: dict[str, Any]) -> StrategyFn:
    """Instantiate the exact pure causal `StrategyFn` from family name and hyperparameters."""
    if family == "below_sma":
        return make_below_sma_buyer(
            window=int(params["window"]),
            tranche_frac=float(params["tranche_frac"]),
        )
    if family == "rsi":
        return make_rsi_buyer(
            window=int(params["window"]),
            threshold=float(params["threshold"]),
            tranche_frac=float(params["tranche_frac"]),
        )
    if family == "bollinger_lower":
        return make_bollinger_lower_buyer(
            window=int(params["window"]),
            num_std=float(params["num_std"]),
            tranche_frac=float(params["tranche_frac"]),
        )
    if family == "vol_scaled_pacing":
        return make_vol_scaled_pacing(
            short_window=int(params["short_window"]),
            long_window=int(params["long_window"]),
            sensitivity=float(params["sensitivity"]),
        )
    if family == "adaptive_dd_pacing":
        return make_adaptive_drawdown_pacing(
            window=int(params["window"]),
            boost_per_10pct_dd=float(params["boost_per_10pct_dd"]),
        )
    raise ValueError(f"Unknown strategy family: {family}")


def evaluate_family_grid_on_mc(
    dates: pd.DatetimeIndex,
    prices: np.ndarray,
    *,
    n_boot_paths: int,
    annual_budget: float = DEFAULT_ASSUMPTIONS.annual_budget,
) -> pd.DataFrame:
    """Grid-search all 5 technical & adaptive families on combined Bootstrap + GJR-GARCH paths."""
    slices = _year_slices(dates)
    lump_term, dca_term = _benchmark_mc_terminals(dates, prices, annual_budget)
    rows: list[dict[str, Any]] = []

    def _record(
        family: str,
        params: dict[str, Any],
        term: np.ndarray,
        avg_buy_day: float,
    ) -> None:
        rel_lump = term / lump_term - 1.0
        rel_dca = term / dca_term - 1.0
        rows.append(
            {
                "family": family,
                "params_label": ", ".join(f"{k}={v}" for k, v in params.items()),
                "params_json": str(params),
                **{f"param_{k}": v for k, v in params.items()},
                "mc_mean_vs_lump_pct": float(np.mean(rel_lump)),
                "mc_boot_mean_vs_lump_pct": float(np.mean(rel_lump[:n_boot_paths])),
                "mc_garch_mean_vs_lump_pct": float(np.mean(rel_lump[n_boot_paths:])),
                "mc_median_vs_lump_pct": float(np.median(rel_lump)),
                "mc_win_rate_vs_lump": float((term > lump_term + 1e-9).mean()),
                "mc_p05_regret_vs_lump_pct": float(np.quantile(rel_lump, 0.05)),
                "mc_mean_vs_dca_pct": float(np.mean(rel_dca)),
                "mc_win_rate_vs_dca": float((term > dca_term + 1e-9).mean()),
                "mc_avg_buy_day": avg_buy_day,
            }
        )

    # 1. Below SMA grid (20 configs)
    for window in (10, 20, 50, 100, 200):
        sma = _compute_sma(prices, window)
        mask = (~np.isnan(sma)) & (prices < sma)
        for tranche in (0.10, 0.25, 0.50, 1.00):
            term, day = _run_signal_tranche_mc(prices, slices, mask, tranche, annual_budget)
            _record("below_sma", {"window": window, "tranche_frac": tranche}, term, day)

    # 2. RSI oversold grid (72 configs)
    for window in (5, 10, 14, 21):
        rsi = _compute_simple_rsi(prices, window)
        for thresh in (25.0, 30.0, 35.0, 40.0, 45.0, 50.0):
            mask = (~np.isnan(rsi)) & (rsi <= thresh)
            for tranche in (0.25, 0.50, 1.00):
                term, day = _run_signal_tranche_mc(prices, slices, mask, tranche, annual_budget)
                _record(
                    "rsi",
                    {"window": window, "threshold": thresh, "tranche_frac": tranche},
                    term,
                    day,
                )

    # 3. Bollinger Lower Band grid (45 configs)
    for window in (10, 20, 50):
        sma = _compute_sma(prices, window)
        std = _compute_rolling_std(prices, window)
        for num_std in (0.5, 1.0, 1.5, 2.0, 2.5):
            mask = (~np.isnan(sma)) & (prices <= sma - num_std * std)
            for tranche in (0.25, 0.50, 1.00):
                term, day = _run_signal_tranche_mc(prices, slices, mask, tranche, annual_budget)
                _record(
                    "bollinger_lower",
                    {"window": window, "num_std": num_std, "tranche_frac": tranche},
                    term,
                    day,
                )

    # 4. Volatility-Scaled Pacing grid (30 configs)
    for short_w in (10, 20, 40):
        for long_w in (126, 252):
            ratio = _compute_vol_ratio_matrix(prices, short_w, long_w)
            for sens in (0.5, 1.0, 1.5, 2.5, 4.0):
                mult = np.clip(np.power(ratio, sens), 0.25, 4.0)
                term, day = _run_pacing_mult_mc(prices, slices, mult, annual_budget)
                _record(
                    "vol_scaled_pacing",
                    {"short_window": short_w, "long_window": long_w, "sensitivity": sens},
                    term,
                    day,
                )

    # 5. Adaptive Drawdown Pacing grid (21 configs)
    for window in (63, 126, 252):
        dd = _compute_trailing_dd_matrix(prices, window)
        abs_dd_scaled = np.abs(dd) / 0.10
        for boost in (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0):
            mult = 1.0 + boost * abs_dd_scaled
            term, day = _run_pacing_mult_mc(prices, slices, mult, annual_budget)
            _record(
                "adaptive_dd_pacing",
                {"window": window, "boost_per_10pct_dd": boost},
                term,
                day,
            )

    return pd.DataFrame(rows)


def _plot_tuning_results(
    grid_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    out_path: Path,
) -> None:
    """Render MC sensitivity heatmaps (top) and Default vs MC-Tuned historical OOS (bottom)."""
    fig = plt.figure(figsize=(15.5, 9.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.1])

    # Top-left: Below SMA heatmap (window vs tranche_frac).
    ax1 = fig.add_subplot(gs[0, 0])
    sub_sma = grid_df[grid_df["family"] == "below_sma"].pivot(
        index="param_window", columns="param_tranche_frac", values="mc_mean_vs_lump_pct"
    )
    im1 = ax1.imshow(sub_sma.to_numpy() * 100.0, cmap="RdYlGn", aspect="auto", vmin=-4.5, vmax=0.5)
    ax1.set_xticks(range(len(sub_sma.columns)))
    ax1.set_xticklabels([f"{c:.2f}" for c in sub_sma.columns], fontsize=8)
    ax1.set_yticks(range(len(sub_sma.index)))
    ax1.set_yticklabels([int(i) for i in sub_sma.index], fontsize=8)
    ax1.set_xlabel("Tranche Fraction")
    ax1.set_ylabel("SMA Window (days)")
    ax1.set_title("Below SMA: MC Mean vs Lump (%)", fontsize=10, fontweight="bold")
    for r in range(sub_sma.shape[0]):
        for c in range(sub_sma.shape[1]):
            ax1.text(
                c,
                r,
                f"{sub_sma.iloc[r, c] * 100:.2f}",
                ha="center",
                va="center",
                fontsize=7.5,
            )
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    # Top-middle: RSI heatmap (threshold vs window at tranche=1.0).
    ax2 = fig.add_subplot(gs[0, 1])
    sub_rsi = grid_df[(grid_df["family"] == "rsi") & (grid_df["param_tranche_frac"] == 1.0)].pivot(
        index="param_window", columns="param_threshold", values="mc_mean_vs_lump_pct"
    )
    im2 = ax2.imshow(sub_rsi.to_numpy() * 100.0, cmap="RdYlGn", aspect="auto", vmin=-4.5, vmax=0.5)
    ax2.set_xticks(range(len(sub_rsi.columns)))
    ax2.set_xticklabels([int(c) for c in sub_rsi.columns], fontsize=8)
    ax2.set_yticks(range(len(sub_rsi.index)))
    ax2.set_yticklabels([int(i) for i in sub_rsi.index], fontsize=8)
    ax2.set_xlabel("RSI Threshold (tranche=1.0)")
    ax2.set_ylabel("RSI Window (days)")
    ax2.set_title("RSI Oversold: MC Mean vs Lump (%)", fontsize=10, fontweight="bold")
    for r in range(sub_rsi.shape[0]):
        for c in range(sub_rsi.shape[1]):
            ax2.text(
                c,
                r,
                f"{sub_rsi.iloc[r, c] * 100:.2f}",
                ha="center",
                va="center",
                fontsize=7.5,
            )
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    # Top-right: Adaptive DD Pacing (window vs boost_per_10pct_dd).
    ax3 = fig.add_subplot(gs[0, 2])
    sub_ad = grid_df[grid_df["family"] == "adaptive_dd_pacing"].pivot(
        index="param_window", columns="param_boost_per_10pct_dd", values="mc_mean_vs_lump_pct"
    )
    im3 = ax3.imshow(sub_ad.to_numpy() * 100.0, cmap="RdYlGn", aspect="auto", vmin=-4.5, vmax=0.5)
    ax3.set_xticks(range(len(sub_ad.columns)))
    ax3.set_xticklabels([int(c) for c in sub_ad.columns], fontsize=8)
    ax3.set_yticks(range(len(sub_ad.index)))
    ax3.set_yticklabels([int(i) for i in sub_ad.index], fontsize=8)
    ax3.set_xlabel("Boost per 10% Drawdown")
    ax3.set_ylabel("Lookback Window (days)")
    ax3.set_title("Adaptive DD Pacing: MC Mean vs Lump (%)", fontsize=10, fontweight="bold")
    for r in range(sub_ad.shape[0]):
        for c in range(sub_ad.shape[1]):
            ax3.text(
                c,
                r,
                f"{sub_ad.iloc[r, c] * 100:.2f}",
                ha="center",
                va="center",
                fontsize=7.5,
            )
    fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    # Bottom spanning all 3 columns: Default vs MC-Tuned on SPY OOS (2016-2025) & QQQ OOS.
    ax4 = fig.add_subplot(gs[1, :])
    families = [
        "below_sma",
        "rsi",
        "bollinger_lower",
        "vol_scaled_pacing",
        "adaptive_dd_pacing",
    ]
    x = np.arange(len(families))
    w = 0.19
    def_rows = (
        comparison_df[comparison_df["variant"] == "default"].set_index("family").loc[families]
    )
    tun_rows = (
        comparison_df[comparison_df["variant"] == "mc_tuned"].set_index("family").loc[families]
    )

    ax4.bar(
        x - 1.5 * w,
        def_rows["spy_oos_vs_lump_pct"] * 100.0,
        width=w,
        label="SPY OOS (2016–2025) — Default",
        color="#aec7e8",
    )
    ax4.bar(
        x - 0.5 * w,
        tun_rows["spy_oos_vs_lump_pct"] * 100.0,
        width=w,
        label="SPY OOS (2016–2025) — MC-Tuned",
        color="#1f77b4",
    )
    ax4.bar(
        x + 0.5 * w,
        def_rows["qqq_oos_vs_lump_pct"] * 100.0,
        width=w,
        label="QQQ OOS (2016–2025) — Default",
        color="#ffbb78",
    )
    ax4.bar(
        x + 1.5 * w,
        tun_rows["qqq_oos_vs_lump_pct"] * 100.0,
        width=w,
        label="QQQ OOS (2016–2025) — MC-Tuned",
        color="#ff7f0e",
    )
    ax4.axhline(0.0, color="#222222", linewidth=1.1, linestyle="--", label="Day-1 Lump Sum (0%)")
    ax4.set_xticks(x)
    ax4.set_xticklabels(families, fontsize=9.5)
    ax4.set_ylabel("Out-of-Sample Terminal Wealth vs Day-1 Lump Sum (%)")
    ax4.set_title(
        "MC-Tuned vs Default Parameters Applied to Held-Out Actual History (2016–2025)",
        fontsize=11,
        fontweight="bold",
    )
    ax4.legend(ncol=3, fontsize=8.5, loc="lower left")
    ax4.grid(axis="y", linestyle="--", alpha=0.35)
    ax4.spines["top"].set_visible(False)
    ax4.spines["right"].set_visible(False)

    fig.suptitle(
        "Monte Carlo Hyperparameter Tuning (Calibrated Strictly on SPY <= 2015) & OOS Validation",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def run_mc_hyperparameter_tuning(
    *,
    n_paths_per_model: int = 500,
    mc_years: int = 10,
    seed: int = 3026,
    output_dir: Path = REPORTS_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tune technical & adaptive strategies on MC data, then test on real historical SPY/QQQ."""
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    spy_raw = load_raw_dataset("SPY")
    qqq_raw = load_raw_dataset("QQQ")
    irx_raw = load_raw_dataset("^IRX")
    rf_series = irx_raw["close"]
    spy_complete = filter_complete_years(spy_raw)
    train_end = DEFAULT_ASSUMPTIONS.train_end_year

    # 1. Generate combined MC tuning set strictly from SPY <= 2015.
    dates, boot_prices, _ = _generate_mc_price_matrix(
        "block_bootstrap",
        train_price_df=spy_complete,
        train_end_year=train_end,
        n_years=mc_years,
        n_paths=n_paths_per_model,
        seed=seed,
    )
    _, garch_prices, _ = _generate_mc_price_matrix(
        "gjr_garch_t",
        train_price_df=spy_complete,
        train_end_year=train_end,
        n_years=mc_years,
        n_paths=n_paths_per_model,
        seed=seed + 1,
    )
    combined_prices = np.vstack([boot_prices, garch_prices])

    # 2. Evaluate full 188-config grid on combined 1,000 synthetic paths.
    grid_df = evaluate_family_grid_on_mc(
        dates,
        combined_prices,
        n_boot_paths=n_paths_per_model,
        annual_budget=DEFAULT_ASSUMPTIONS.annual_budget,
    )
    grid_df.to_csv(tables_dir / "mc_tuning_grid.csv", index=False)

    # 3. Extract default vs MC-optimal config per family.
    default_params: dict[str, dict[str, Any]] = {
        "below_sma": {"window": 200, "tranche_frac": 0.10},
        "rsi": {"window": 14, "threshold": 30.0, "tranche_frac": 0.50},
        "bollinger_lower": {"window": 20, "num_std": 2.0, "tranche_frac": 0.50},
        "vol_scaled_pacing": {"short_window": 20, "long_window": 252, "sensitivity": 1.5},
        "adaptive_dd_pacing": {"window": 252, "boost_per_10pct_dd": 4.0},
    }

    # Precompute historical benchmark references.
    benchmarks: dict[str, dict[str, float]] = {}
    for key, df, s_yr, rf in (
        ("spy_oos", spy_raw, train_end + 1, None),
        ("spy_full", spy_raw, None, None),
        ("spy_full_irx", spy_raw, None, rf_series),
        ("qqq_oos", qqq_raw, train_end + 1, None),
        ("qqq_full", qqq_raw, None, None),
    ):
        lump_s = summarize_backtest(
            run_backtest(df, lump_sum_day1, start_year=s_yr, cash_yield_series=rf)
        )
        dca_s = summarize_backtest(
            run_backtest(df, make_monthly_dca(1), start_year=s_yr, cash_yield_series=rf)
        )
        benchmarks[key] = {
            "lump_term": lump_s["terminal_value"],
            "dca_term": dca_s["terminal_value"],
        }

    comp_rows: list[dict[str, Any]] = []
    for family in (
        "below_sma",
        "rsi",
        "bollinger_lower",
        "vol_scaled_pacing",
        "adaptive_dd_pacing",
    ):
        fam_df = grid_df[grid_df["family"] == family]
        best_row = fam_df.loc[fam_df["mc_mean_vs_lump_pct"].idxmax()]
        best_params = {
            col.removeprefix("param_"): (
                int(best_row[col])
                if col in ("param_window", "param_short_window", "param_long_window")
                else float(best_row[col])
            )
            for col in fam_df.columns
            if col.startswith("param_") and pd.notna(best_row[col])
        }

        # Find default row in grid for MC stats.
        def_p = default_params[family]
        def_label = ", ".join(f"{k}={v}" for k, v in def_p.items())
        def_row = fam_df[fam_df["params_label"] == def_label].iloc[0]

        for variant, params, mc_row in (
            ("default", def_p, def_row),
            ("mc_tuned", best_params, best_row),
        ):
            fn = build_strategy_from_params(family, params)
            # Mechanically verify zero look-ahead on real SPY OOS before recording.
            assert_prefix_invariance(fn, spy_raw[spy_raw.index.year >= 2022])

            row_out: dict[str, Any] = {
                "family": family,
                "variant": variant,
                "params": ", ".join(f"{k}={v}" for k, v in params.items()),
                "mc_mean_vs_lump_pct": float(mc_row["mc_mean_vs_lump_pct"]),
                "mc_mean_vs_dca_pct": float(mc_row["mc_mean_vs_dca_pct"]),
                "mc_win_rate_vs_lump": float(mc_row["mc_win_rate_vs_lump"]),
                "mc_win_rate_vs_dca": float(mc_row["mc_win_rate_vs_dca"]),
                "mc_avg_buy_day": float(mc_row["mc_avg_buy_day"]),
            }

            for key, df, s_yr, rf in (
                ("spy_oos", spy_raw, train_end + 1, None),
                ("spy_full", spy_raw, None, None),
                ("spy_full_irx", spy_raw, None, rf_series),
                ("qqq_oos", qqq_raw, train_end + 1, None),
                ("qqq_full", qqq_raw, None, None),
            ):
                res = run_backtest(
                    df,
                    fn,
                    strategy_name=f"{family}_{variant}",
                    start_year=s_yr,
                    cash_yield_series=rf,
                )
                summ = summarize_backtest(res)
                row_out[f"{key}_vs_lump_pct"] = (
                    summ["terminal_value"] / benchmarks[key]["lump_term"] - 1.0
                )
                row_out[f"{key}_vs_dca_pct"] = (
                    summ["terminal_value"] / benchmarks[key]["dca_term"] - 1.0
                )
                if key == "spy_full":
                    row_out["spy_full_xirr"] = summ["xirr"]
                    row_out["spy_full_forced_sweep_pct"] = summ["avg_forced_sweep_pct"]
                    row_out["spy_full_avg_buy_day"] = summ["avg_weighted_buy_day"]

            comp_rows.append(row_out)

    comp_df = pd.DataFrame(comp_rows)
    comp_df.to_csv(tables_dir / "mc_tuned_vs_historical.csv", index=False)
    _plot_tuning_results(grid_df, comp_df, figures_dir / "07_mc_hyperparameter_tuning.png")
    return grid_df, comp_df


def main() -> None:
    """CLI entrypoint to run MC hyperparameter tuning and historical OOS evaluation."""
    grid_df, comp_df = run_mc_hyperparameter_tuning()
    print(f"Evaluated {len(grid_df)} parameter configs on 1,000 synthetic MC paths.")
    cols = [
        "family",
        "variant",
        "params",
        "mc_mean_vs_lump_pct",
        "spy_oos_vs_lump_pct",
        "spy_full_vs_lump_pct",
        "qqq_oos_vs_lump_pct",
        "qqq_full_vs_lump_pct",
        "spy_full_avg_buy_day",
    ]
    print(comp_df[cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
