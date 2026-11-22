"""Full historical, rolling, regime, cash-yield, synthetic Monte Carlo, and stress analysis."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from etf_strategy.config import DEFAULT_ASSUMPTIONS, ModelingAssumptions
from etf_strategy.data import DEFAULT_MANIFEST_PATH, filter_complete_years, load_raw_dataset
from etf_strategy.engine import BacktestResult, run_backtest
from etf_strategy.metrics import summarize_backtest
from etf_strategy.strategies import (
    StrategySpec,
    build_default_strategy_registry,
    make_monthly_dca,
    make_oracle_strategy,
)
from etf_strategy.synthetic import (
    compute_stylized_facts,
    extract_log_returns,
    fit_gjr_garch_t,
    generate_stress_archetype,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = REPO_ROOT / "reports"
TABLES_DIR = REPORTS_DIR / "tables"
FIGURES_DIR = REPORTS_DIR / "figures"


# ---------------------------------------------------------------------------
# Historical & Sensitivity Analysis (Exact Engine)
# ---------------------------------------------------------------------------


def run_strategy_suite(
    price_df: pd.DataFrame,
    *,
    assumptions: ModelingAssumptions = DEFAULT_ASSUMPTIONS,
    cash_yield_series: pd.Series | None = None,
    start_year: int | None = None,
    end_year: int | None = None,
    include_oracles: bool = True,
) -> tuple[pd.DataFrame, dict[str, BacktestResult]]:
    """Run all registered causal strategies (and optional oracles) and return summary table."""
    specs: list[StrategySpec] = []
    if include_oracles:
        specs.append(make_oracle_strategy(price_df, mode="best"))
    specs.extend(build_default_strategy_registry(annual_budget=assumptions.annual_budget))
    if include_oracles:
        specs.append(make_oracle_strategy(price_df, mode="worst"))

    results: dict[str, BacktestResult] = {}
    rows: list[dict[str, Any]] = []

    for spec in specs:
        res = run_backtest(
            price_df,
            spec.fn,
            strategy_name=spec.name,
            assumptions=assumptions,
            cash_yield_series=cash_yield_series,
            start_year=start_year,
            end_year=end_year,
            is_causal=spec.is_causal,
        )
        results[spec.name] = res
        summary = summarize_backtest(res)
        rows.append(
            {
                "strategy": spec.name,
                "category": spec.category,
                "is_causal": spec.is_causal,
                **summary,
            }
        )

    table = pd.DataFrame(rows).set_index("strategy")

    # Add benchmark-relative metrics vs Lump Sum (Day 1) and Monthly DCA (Day 1).
    lump_res = results["lump_sum_day1"]
    dca_res = results["monthly_dca_day1"]
    lump_term = float(table.loc["lump_sum_day1", "terminal_value"])
    dca_term = float(table.loc["monthly_dca_day1", "terminal_value"])
    lump_shares = lump_res.annual["shares_bought"]
    dca_shares = dca_res.annual["shares_bought"]

    vs_lump_pct: list[float] = []
    vs_dca_pct: list[float] = []
    win_vs_lump: list[float] = []
    win_vs_dca: list[float] = []
    worst_yr_vs_lump: list[float] = []
    worst_yr_vs_dca: list[float] = []

    for name in table.index:
        res = results[name]
        shares = res.annual["shares_bought"]
        rel_lump = (shares / lump_shares) - 1.0
        rel_dca = (shares / dca_shares) - 1.0

        vs_lump_pct.append(float(table.loc[name, "terminal_value"]) / lump_term - 1.0)
        vs_dca_pct.append(float(table.loc[name, "terminal_value"]) / dca_term - 1.0)
        win_vs_lump.append(float((shares > lump_shares + 1e-10).mean()))
        win_vs_dca.append(float((shares > dca_shares + 1e-10).mean()))
        worst_yr_vs_lump.append(float(rel_lump.min()))
        worst_yr_vs_dca.append(float(rel_dca.min()))

    table["diff_vs_lump_pct"] = vs_lump_pct
    table["diff_vs_dca_pct"] = vs_dca_pct
    table["annual_win_rate_vs_lump"] = win_vs_lump
    table["annual_win_rate_vs_dca"] = win_vs_dca
    table["worst_year_vs_lump_pct"] = worst_yr_vs_lump
    table["worst_year_vs_dca_pct"] = worst_yr_vs_dca
    return table.sort_values("terminal_value", ascending=False), results


def compute_rolling_window_summary(
    results: dict[str, BacktestResult],
    window_years: int,
) -> pd.DataFrame:
    """Evaluate rolling W-year terminal wealth using annual share accumulation.

    Under buy-only DRIP with fixed annual budget, shares bought in year y are strictly
    independent of prior years' positions, so terminal value over any window [y0, y1] is
    exactly `year_end_price[y1] * sum(shares_bought[y0..y1])`.
    """
    lump_annual = results["lump_sum_day1"].annual
    dca_annual = results["monthly_dca_day1"].annual
    years = list(lump_annual.index)
    if len(years) < window_years:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for name, res in results.items():
        if not res.is_causal:
            continue
        shares = res.annual["shares_bought"].to_numpy(dtype=float)
        lump_s = lump_annual["shares_bought"].to_numpy(dtype=float)
        dca_s = dca_annual["shares_bought"].to_numpy(dtype=float)

        rel_lump_list: list[float] = []
        rel_dca_list: list[float] = []
        for i in range(len(years) - window_years + 1):
            w_s = float(np.sum(shares[i : i + window_years]))
            w_l = float(np.sum(lump_s[i : i + window_years]))
            w_d = float(np.sum(dca_s[i : i + window_years]))
            rel_lump_list.append(w_s / w_l - 1.0)
            rel_dca_list.append(w_s / w_d - 1.0)

        arr_l = np.array(rel_lump_list, dtype=float)
        arr_d = np.array(rel_dca_list, dtype=float)
        rows.append(
            {
                "strategy": name,
                "window_years": window_years,
                "n_windows": len(arr_l),
                "win_rate_vs_lump": float((arr_l > 1e-10).mean()),
                "median_vs_lump_pct": float(np.median(arr_l)),
                "mean_vs_lump_pct": float(np.mean(arr_l)),
                "worst_vs_lump_pct": float(np.min(arr_l)),
                "win_rate_vs_dca": float((arr_d > 1e-10).mean()),
                "median_vs_dca_pct": float(np.median(arr_d)),
                "mean_vs_dca_pct": float(np.mean(arr_d)),
                "worst_vs_dca_pct": float(np.min(arr_d)),
            }
        )
    return pd.DataFrame(rows).set_index("strategy").sort_values("mean_vs_lump_pct", ascending=False)


def compute_regime_breakdown(
    price_df: pd.DataFrame,
    results: dict[str, BacktestResult],
    *,
    price_col: str = DEFAULT_ASSUMPTIONS.execution_price_col,
) -> pd.DataFrame:
    """Break down mean annual share-accumulation edge vs Lump Sum & Monthly DCA by market regime."""
    lump_annual = results["lump_sum_day1"].annual
    dca_annual = results["monthly_dca_day1"].annual
    years = list(lump_annual.index)

    # Classify each calendar year by total return (first trading day to last trading day).
    regime_map: dict[int, str] = {}
    for year in years:
        yr_prices = price_df[price_df.index.year == year][price_col]
        yr_ret = float(yr_prices.iloc[-1] / yr_prices.iloc[0]) - 1.0
        if yr_ret >= 0.15:
            regime_map[year] = "strong_bull (>=15%)"
        elif yr_ret >= 0.0:
            regime_map[year] = "moderate_bull (0-15%)"
        else:
            regime_map[year] = "bear (<0%)"

    regimes = pd.Series(regime_map, name="regime")
    rows: list[dict[str, Any]] = []
    for name, res in results.items():
        if not res.is_causal:
            continue
        rel_lump = (res.annual["shares_bought"] / lump_annual["shares_bought"]) - 1.0
        rel_dca = (res.annual["shares_bought"] / dca_annual["shares_bought"]) - 1.0
        for regime_name in ("strong_bull (>=15%)", "moderate_bull (0-15%)", "bear (<0%)"):
            mask = regimes == regime_name
            if not mask.any():
                continue
            rows.append(
                {
                    "strategy": name,
                    "regime": regime_name,
                    "n_years": int(mask.sum()),
                    "mean_edge_vs_lump_pct": float(rel_lump[mask].mean()),
                    "mean_edge_vs_dca_pct": float(rel_dca[mask].mean()),
                    "win_rate_vs_lump": float((rel_lump[mask] > 1e-10).mean()),
                    "win_rate_vs_dca": float((rel_dca[mask] > 1e-10).mean()),
                }
            )
    return pd.DataFrame(rows)


def run_monthly_day_sensitivity(
    price_df: pd.DataFrame,
    *,
    assumptions: ModelingAssumptions = DEFAULT_ASSUMPTIONS,
) -> pd.DataFrame:
    """Evaluate Monthly Fixed-Day DCA across calendar days 1..28."""
    rows: list[dict[str, Any]] = []
    for day in range(1, 29):
        res = run_backtest(
            price_df,
            make_monthly_dca(day),
            strategy_name=f"monthly_day_{day:02d}",
            assumptions=assumptions,
        )
        summary = summarize_backtest(res)
        rows.append({"day_of_month": day, **summary})
    return pd.DataFrame(rows).set_index("day_of_month")


# ---------------------------------------------------------------------------
# Fast Vectorized Monte Carlo Evaluator (Verified Against Exact Engine)
# ---------------------------------------------------------------------------


def _generate_mc_price_matrix(
    model: str,
    *,
    train_price_df: pd.DataFrame,
    train_end_year: int,
    n_years: int,
    n_paths: int,
    seed: int,
) -> tuple[pd.DatetimeIndex, np.ndarray, dict[str, Any]]:
    """Generate `(n_paths, n_days)` simulated price matrix for fast N=1,000+ Monte Carlo."""
    dates = pd.bdate_range(start="2001-01-01", end=f"{2001 + n_years - 1}-12-31")
    n_steps = len(dates)
    rng = np.random.default_rng(seed)
    hist_returns = extract_log_returns(train_price_df, train_end_year=train_end_year)
    meta: dict[str, Any] = {
        "model": model,
        "seed": seed,
        "n_years": n_years,
        "n_paths": n_paths,
        "train_end_year": train_end_year,
    }

    if model == "block_bootstrap":
        expected_block_length = 20.0
        p_new = 1.0 / expected_block_length
        n_hist = len(hist_returns)
        idx = np.empty((n_paths, n_steps), dtype=np.int64)
        idx[:, 0] = rng.integers(0, n_hist, size=n_paths)
        switches = rng.random((n_paths, n_steps)) < p_new
        starts = rng.integers(0, n_hist, size=(n_paths, n_steps))
        for t in range(1, n_steps):
            idx[:, t] = np.where(switches[:, t], starts[:, t], (idx[:, t - 1] + 1) % n_hist)
        log_ret = hist_returns[idx]
        meta["expected_block_length"] = expected_block_length
    elif model == "gjr_garch_t":
        params = fit_gjr_garch_t(train_price_df, train_end_year=train_end_year)
        burn_in = 250
        total_steps = n_steps + burn_in
        t_scale = math.sqrt((params.nu - 2.0) / params.nu)
        z = rng.standard_t(df=params.nu, size=(n_paths, total_steps)) * t_scale
        h = np.full(n_paths, params.unconditional_variance, dtype=float)
        eps = np.zeros(n_paths, dtype=float)
        all_ret = np.empty((n_paths, total_steps), dtype=float)
        for t in range(total_steps):
            if t > 0:
                neg = (eps < 0.0).astype(float)
                h = (
                    params.omega
                    + (params.alpha + params.gamma * neg) * (eps * eps)
                    + params.beta * h
                )
            eps = np.sqrt(np.maximum(h, 1e-18)) * z[:, t]
            all_ret[:, t] = params.mu + eps
        log_ret = all_ret[:, burn_in:]
        meta["params"] = {
            "mu": params.mu,
            "omega": params.omega,
            "alpha": params.alpha,
            "gamma": params.gamma,
            "beta": params.beta,
            "nu": params.nu,
            "persistence": params.persistence,
        }
    elif model == "gbm_null":
        ann_drift = float(np.mean(hist_returns) * 252.0)
        ann_vol = float(np.std(hist_returns, ddof=1) * math.sqrt(252.0))
        dt = 1.0 / 252.0
        # Match log-return drift directly so geometric mean matches empirical training drift.
        log_ret = (ann_drift * dt) + (ann_vol * math.sqrt(dt)) * rng.standard_normal(
            (n_paths, n_steps)
        )
        meta["annual_log_drift"] = ann_drift
        meta["annual_vol"] = ann_vol
    else:
        raise ValueError(f"Unknown Monte Carlo model: {model}")

    prices = 100.0 * np.exp(np.cumsum(log_ret, axis=1))
    return dates, prices, meta


def evaluate_vectorized_mc_strategies(
    dates: pd.DatetimeIndex,
    prices: np.ndarray,
    *,
    annual_budget: float = DEFAULT_ASSUMPTIONS.annual_budget,
) -> dict[str, np.ndarray]:
    """Vectorized exact-equivalent evaluation of core strategies on `(n_paths, n_days)` prices.

    Returns a mapping `strategy_name -> terminal_value_array (shape: n_paths)`.
    Equivalence with `run_backtest` is verified in `tests/test_analysis.py`.
    """
    n_paths, n_days = prices.shape
    years = dates.year.to_numpy()
    months = dates.month.to_numpy()
    unique_years = sorted({int(y) for y in years})

    # Precompute year slices.
    year_slices: list[tuple[int, int]] = []
    for y in unique_years:
        idx = np.flatnonzero(years == y)
        year_slices.append((int(idx[0]), int(idx[-1]) + 1))

    # Precompute strictly causal indicators across all paths simultaneously.
    # 1. Trailing 252-day max & 50/200-day SMAs & 14-day simple RSI.
    sma50 = np.full_like(prices, np.nan)
    sma200 = np.full_like(prices, np.nan)
    csum = np.cumsum(prices, axis=1)
    sma50[:, 49] = csum[:, 49] / 50.0
    sma50[:, 50:] = (csum[:, 50:] - csum[:, :-50]) / 50.0
    if n_days >= 200:
        sma200[:, 199] = csum[:, 199] / 200.0
        sma200[:, 200:] = (csum[:, 200:] - csum[:, :-200]) / 200.0

    # Trailing 14-day simple RSI matching `make_rsi_buyer`.
    rsi14 = np.full_like(prices, np.nan)
    deltas = np.diff(prices, axis=1)
    gains = np.maximum(deltas, 0.0)
    losses = np.maximum(-deltas, 0.0)
    cg = np.cumsum(gains, axis=1)
    cl = np.cumsum(losses, axis=1)
    avg_g = np.empty((n_paths, n_days - 14), dtype=float)
    avg_l = np.empty((n_paths, n_days - 14), dtype=float)
    avg_g[:, 0] = cg[:, 13] / 14.0
    avg_g[:, 1:] = (cg[:, 14:] - cg[:, :-14]) / 14.0
    avg_l[:, 0] = cl[:, 13] / 14.0
    avg_l[:, 1:] = (cl[:, 14:] - cl[:, :-14]) / 14.0
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = np.where(avg_l == 0.0, np.inf, avg_g / avg_l)
        rsi14[:, 14:] = np.where(avg_l == 0.0, 100.0, 100.0 - (100.0 / (1.0 + rs)))

    # Trailing 252-day rolling high (causal).
    roll252_high = np.empty_like(prices)
    for t in range(n_days):
        lo = max(0, t - 251)
        roll252_high[:, t] = np.max(prices[:, lo : t + 1], axis=1)
    dd252 = (prices / roll252_high) - 1.0

    # Calendar trigger masks.
    is_new_month = np.ones(n_days, dtype=bool)
    is_new_month[1:] = (months[1:] != months[:-1]) | (years[1:] != years[:-1])

    strategy_names = (
        "lump_sum_day1",
        "monthly_dca_day1",
        "quarterly_dca",
        "front_loaded_q1",
        "dip_ytd_5pct",
        "dip_ytd_10pct",
        "dip_tiered_5_10_15",
        "hybrid_50_lump_dip10",
        "below_sma50_all_in",
        "below_sma200_paced",
        "rsi14_oversold_30",
        "adaptive_dd_pacing",
    )
    shares: dict[str, np.ndarray] = {
        name: np.zeros(n_paths, dtype=float) for name in strategy_names
    }

    for start_i, end_i in year_slices:
        year_len = end_i - start_i
        rem = {name: np.full(n_paths, annual_budget, dtype=float) for name in strategy_names}
        ytd_high = prices[:, start_i].copy()

        for offset in range(year_len):
            t = start_i + offset
            days_left = year_len - offset
            p_t = prices[:, t]
            inv_p = 1.0 / p_t
            ytd_high = np.maximum(ytd_high, p_t)
            dd_ytd = (p_t / ytd_high) - 1.0
            dd_252_t = dd252[:, t]
            m = int(months[t])
            new_m = bool(is_new_month[t])

            # Compute requested spend per strategy.
            req: dict[str, np.ndarray] = {}
            req["lump_sum_day1"] = rem["lump_sum_day1"].copy()

            if new_m:
                req["monthly_dca_day1"] = rem["monthly_dca_day1"] / (13 - m)
            else:
                req["monthly_dca_day1"] = np.zeros(n_paths, dtype=float)

            q_map = {1: 4, 4: 3, 7: 2, 10: 1}
            if new_m and m in q_map:
                req["quarterly_dca"] = rem["quarterly_dca"] / q_map[m]
            else:
                req["quarterly_dca"] = np.zeros(n_paths, dtype=float)

            q1_map = {1: 3, 2: 2, 3: 1}
            if new_m and m in q1_map:
                req["front_loaded_q1"] = rem["front_loaded_q1"] / q1_map[m]
            else:
                req["front_loaded_q1"] = np.zeros(n_paths, dtype=float)

            req["dip_ytd_5pct"] = np.where(dd_ytd <= -0.05, rem["dip_ytd_5pct"], 0.0)
            req["dip_ytd_10pct"] = np.where(dd_ytd <= -0.10, rem["dip_ytd_10pct"], 0.0)

            target_dep = np.where(
                dd_252_t <= -0.15,
                1.0,
                np.where(dd_252_t <= -0.10, 0.66, np.where(dd_252_t <= -0.05, 0.33, 0.0)),
            )
            target_rem = annual_budget * (1.0 - target_dep)
            req["dip_tiered_5_10_15"] = np.where(
                target_dep > 0.0,
                np.minimum(
                    rem["dip_tiered_5_10_15"],
                    np.maximum(0.0, rem["dip_tiered_5_10_15"] - target_rem),
                ),
                0.0,
            )

            if offset == 0:
                req["hybrid_50_lump_dip10"] = 0.5 * rem["hybrid_50_lump_dip10"]
            else:
                req["hybrid_50_lump_dip10"] = np.where(
                    dd_252_t <= -0.10, rem["hybrid_50_lump_dip10"], 0.0
                )

            req["below_sma50_all_in"] = np.where(
                (~np.isnan(sma50[:, t])) & (p_t < sma50[:, t]), rem["below_sma50_all_in"], 0.0
            )
            req["below_sma200_paced"] = np.where(
                (~np.isnan(sma200[:, t])) & (p_t < sma200[:, t]),
                0.10 * rem["below_sma200_paced"],
                0.0,
            )
            req["rsi14_oversold_30"] = np.where(
                (~np.isnan(rsi14[:, t])) & (rsi14[:, t] <= 30.0),
                0.50 * rem["rsi14_oversold_30"],
                0.0,
            )

            base_pace = rem["adaptive_dd_pacing"] / days_left
            mult = 1.0 + 4.0 * (np.abs(dd_252_t) / 0.10)
            req["adaptive_dd_pacing"] = np.minimum(rem["adaptive_dd_pacing"], base_pace * mult)

            for name in strategy_names:
                spend = rem[name] if days_left == 1 else req[name]
                shares[name] += spend * inv_p
                rem[name] = np.maximum(rem[name] - spend, 0.0)

    final_prices = prices[:, -1]
    return {name: sh * final_prices for name, sh in shares.items()}


def summarize_mc_results(terminal_map: dict[str, np.ndarray]) -> pd.DataFrame:
    """Summarize Monte Carlo terminal wealth distributions relative to benchmarks."""
    lump = terminal_map["lump_sum_day1"]
    dca = terminal_map["monthly_dca_day1"]
    rows: list[dict[str, Any]] = []
    for name, vals in terminal_map.items():
        rel_lump = vals / lump - 1.0
        rel_dca = vals / dca - 1.0
        rows.append(
            {
                "strategy": name,
                "mean_terminal": float(np.mean(vals)),
                "median_terminal": float(np.median(vals)),
                "p10_terminal": float(np.quantile(vals, 0.10)),
                "p90_terminal": float(np.quantile(vals, 0.90)),
                "mean_vs_lump_pct": float(np.mean(rel_lump)),
                "median_vs_lump_pct": float(np.median(rel_lump)),
                "win_rate_vs_lump": float((vals > lump + 1e-9).mean()),
                "p05_regret_vs_lump_pct": float(np.quantile(rel_lump, 0.05)),
                "mean_vs_dca_pct": float(np.mean(rel_dca)),
                "win_rate_vs_dca": float((vals > dca + 1e-9).mean()),
                "p05_regret_vs_dca_pct": float(np.quantile(rel_dca, 0.05)),
            }
        )
    return pd.DataFrame(rows).set_index("strategy").sort_values("mean_terminal", ascending=False)


# ---------------------------------------------------------------------------
# Full Pipeline Runner
# ---------------------------------------------------------------------------


def run_full_study(
    *,
    mc_paths: int = 1_000,
    mc_years: int = 10,
    seed: int = 2026,
    output_dir: Path = REPORTS_DIR,
) -> dict[str, Any]:
    """Execute the complete historical, sensitivity, synthetic MC, and stress-test study."""
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    spy_raw = load_raw_dataset("SPY")
    qqq_raw = load_raw_dataset("QQQ")
    irx_raw = load_raw_dataset("^IRX")
    rf_series = irx_raw["close"]

    # 1. Full historical leaderboards (0% baseline cash yield).
    spy_full, spy_res = run_strategy_suite(spy_raw)
    qqq_full, qqq_res = run_strategy_suite(qqq_raw)
    spy_full.to_csv(tables_dir / "spy_historical_full.csv")
    qqq_full.to_csv(tables_dir / "qqq_historical_full.csv")

    # 2. Chronological train (<=2015) vs OOS (2016-2025) split.
    train_end = DEFAULT_ASSUMPTIONS.train_end_year
    spy_train, _ = run_strategy_suite(spy_raw, end_year=train_end)
    spy_oos, _ = run_strategy_suite(spy_raw, start_year=train_end + 1)
    qqq_train, _ = run_strategy_suite(qqq_raw, end_year=train_end)
    qqq_oos, _ = run_strategy_suite(qqq_raw, start_year=train_end + 1)

    spy_train.to_csv(tables_dir / "spy_train_to_2015.csv")
    spy_oos.to_csv(tables_dir / "spy_oos_2016_2025.csv")
    qqq_train.to_csv(tables_dir / "qqq_train_to_2015.csv")
    qqq_oos.to_csv(tables_dir / "qqq_oos_2016_2025.csv")

    # 3. Rolling multi-year windows (5y, 10y, 20y).
    rolling_tables: dict[str, pd.DataFrame] = {}
    for ticker, res_map in (("spy", spy_res), ("qqq", qqq_res)):
        for w in (5, 10, 20):
            rt = compute_rolling_window_summary(res_map, w)
            if not rt.empty:
                key = f"{ticker}_rolling_{w}y"
                rolling_tables[key] = rt
                rt.to_csv(tables_dir / f"{key}.csv")

    # 4. Regime breakdown.
    spy_regimes = compute_regime_breakdown(spy_raw, spy_res)
    qqq_regimes = compute_regime_breakdown(qqq_raw, qqq_res)
    spy_regimes.to_csv(tables_dir / "spy_regimes.csv", index=False)
    qqq_regimes.to_csv(tables_dir / "qqq_regimes.csv", index=False)

    # 5. Intra-year ^IRX T-bill cash yield sensitivity (full + high-rate 2022-2025).
    spy_rf, _ = run_strategy_suite(spy_raw, cash_yield_series=rf_series)
    qqq_rf, _ = run_strategy_suite(qqq_raw, cash_yield_series=rf_series)
    spy_rf_hi, _ = run_strategy_suite(spy_raw, cash_yield_series=rf_series, start_year=2022)
    spy_0_hi, _ = run_strategy_suite(spy_raw, start_year=2022)
    spy_rf.to_csv(tables_dir / "spy_with_tbill_yield.csv")
    qqq_rf.to_csv(tables_dir / "qqq_with_tbill_yield.csv")
    spy_rf_hi.to_csv(tables_dir / "spy_2022_2025_with_tbill.csv")
    spy_0_hi.to_csv(tables_dir / "spy_2022_2025_zero_yield.csv")

    # 6. Monthly DCA day-of-month sensitivity (days 1..28).
    spy_dom = run_monthly_day_sensitivity(spy_raw)
    qqq_dom = run_monthly_day_sensitivity(qqq_raw)
    spy_dom.to_csv(tables_dir / "spy_monthly_day_sensitivity.csv")
    qqq_dom.to_csv(tables_dir / "qqq_monthly_day_sensitivity.csv")

    # 7. Synthetic Monte Carlo (N=1,000 paths x 10 years, calibrated strictly on SPY <= 2015).
    spy_complete = filter_complete_years(spy_raw)
    mc_summaries: dict[str, pd.DataFrame] = {}
    mc_raw_rel_lump: dict[str, dict[str, np.ndarray]] = {}
    mc_metadata: dict[str, dict[str, Any]] = {}
    for offset_idx, model in enumerate(("block_bootstrap", "gjr_garch_t", "gbm_null")):
        dates, prices, meta = _generate_mc_price_matrix(
            model,
            train_price_df=spy_complete,
            train_end_year=train_end,
            n_years=mc_years,
            n_paths=mc_paths,
            seed=seed + offset_idx,
        )
        term_map = evaluate_vectorized_mc_strategies(dates, prices)
        summary = summarize_mc_results(term_map)
        mc_summaries[model] = summary
        mc_metadata[model] = meta
        lump_vals = term_map["lump_sum_day1"]
        mc_raw_rel_lump[model] = {k: (v / lump_vals - 1.0) for k, v in term_map.items()}
        summary.to_csv(tables_dir / f"mc_{model}_summary.csv")

    # 8. Parametric Stress Archetypes (4 adversarial scenarios).
    stress_tables: dict[str, pd.DataFrame] = {}
    for s_idx, archetype in enumerate(
        ("steady_melt_up", "v_shaped_flash_crash", "slow_bleed_bear", "lost_decade")
    ):
        df_stress = generate_stress_archetype(archetype, seed=seed + 100 + s_idx)
        st_table, _ = run_strategy_suite(df_stress, include_oracles=True)
        stress_tables[archetype] = st_table
        st_table.to_csv(tables_dir / f"stress_{archetype}.csv")

    # Save audit metadata (manifest hashes + seeds + empirical stylized facts).
    manifest = json.loads(DEFAULT_MANIFEST_PATH.read_text(encoding="utf-8"))
    audit = {
        "seed": seed,
        "mc_paths": mc_paths,
        "mc_years": mc_years,
        "train_end_year": train_end,
        "manifest_hashes": {k: v["sha256"] for k, v in manifest.get("datasets", {}).items()},
        "spy_train_stylized_facts": compute_stylized_facts(
            spy_complete[spy_complete.index.year <= train_end]
        ),
        "mc_metadata": mc_metadata,
    }
    (tables_dir / "study_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )

    return {
        "spy_full": spy_full,
        "qqq_full": qqq_full,
        "spy_train": spy_train,
        "spy_oos": spy_oos,
        "qqq_train": qqq_train,
        "qqq_oos": qqq_oos,
        "rolling": rolling_tables,
        "spy_regimes": spy_regimes,
        "qqq_regimes": qqq_regimes,
        "spy_rf": spy_rf,
        "qqq_rf": qqq_rf,
        "spy_rf_hi": spy_rf_hi,
        "spy_0_hi": spy_0_hi,
        "spy_dom": spy_dom,
        "qqq_dom": qqq_dom,
        "mc_summaries": mc_summaries,
        "mc_raw_rel_lump": mc_raw_rel_lump,
        "stress_tables": stress_tables,
        "audit": audit,
    }
