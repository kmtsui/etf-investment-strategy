"""Publication-quality visualization suite for the annual ETF strategy study."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from etf_strategy.analysis import FIGURES_DIR, run_full_study

CATEGORY_COLORS = {
    "oracle": "#6c757d",
    "benchmark": "#1f77b4",
    "calendar": "#2ca02c",
    "drawdown": "#d62728",
    "technical": "#9467bd",
    "adaptive": "#ff7f0e",
}


def _style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle="--", alpha=0.35)


def plot_historical_leaderboard(
    spy_full: pd.DataFrame,
    qqq_full: pd.DataFrame,
    out_path: Path,
) -> None:
    """Side-by-side terminal wealth bars for SPY (1994-2025) and QQQ (2000-2025)."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.5))
    for ax, df, title in (
        (axes[0], spy_full, "SPY (1994–2025, 32 yrs, $384k contributed)"),
        (axes[1], qqq_full, "QQQ (2000–2025, 26 yrs, $312k contributed)"),
    ):
        ordered = df.sort_values("terminal_value", ascending=True)
        colors = [CATEGORY_COLORS.get(c, "#333333") for c in ordered["category"]]
        vals_m = ordered["terminal_value"] / 1e6
        bars = ax.barh(ordered.index, vals_m, color=colors, alpha=0.9)

        lump_m = float(df.loc["lump_sum_day1", "terminal_value"]) / 1e6
        dca_m = float(df.loc["monthly_dca_day1", "terminal_value"]) / 1e6
        ax.axvline(lump_m, color="#1f77b4", linestyle="--", linewidth=1.3, label="Lump Sum Day 1")
        ax.axvline(dca_m, color="#17becf", linestyle=":", linewidth=1.5, label="Monthly DCA Day 1")

        for bar, xirr in zip(bars, ordered["xirr"], strict=True):
            w = bar.get_width()
            ax.text(
                w + 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"${w:.2f}M ({xirr * 100:.2f}%)",
                va="center",
                fontsize=8,
            )

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Terminal Portfolio Value ($ Millions)")
        ax.set_xlim(0, float(vals_m.max()) * 1.28)
        _style_axes(ax)
        ax.legend(loc="lower right", fontsize=8)

    fig.suptitle(
        "Historical Strategy Leaderboard (0% Cash Yield, 100% DRIP, Full Annual Spend)",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_insample_vs_oos(
    spy_train: pd.DataFrame,
    spy_oos: pd.DataFrame,
    qqq_train: pd.DataFrame,
    qqq_oos: pd.DataFrame,
    out_path: Path,
) -> None:
    """Scatter of In-Sample (<=2015) vs Out-of-Sample (2016-2025) excess wealth vs Lump Sum."""
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8))
    for ax, tr, oos, title in (
        (axes[0], spy_train, spy_oos, "SPY: Train (1994–2015) vs OOS (2016–2025)"),
        (axes[1], qqq_train, qqq_oos, "QQQ: Train (2000–2015) vs OOS (2016–2025)"),
    ):
        causal = tr[tr["is_causal"]].index
        x = tr.loc[causal, "diff_vs_lump_pct"] * 100.0
        y = oos.loc[causal, "diff_vs_lump_pct"] * 100.0
        cats = tr.loc[causal, "category"]

        ax.axhline(0.0, color="#444444", linewidth=1.0, linestyle="--")
        ax.axvline(0.0, color="#444444", linewidth=1.0, linestyle="--")

        for name in causal:
            ax.scatter(
                x.loc[name],
                y.loc[name],
                color=CATEGORY_COLORS.get(cats.loc[name], "#333333"),
                s=65,
                alpha=0.9,
                edgecolors="white",
                linewidth=0.6,
            )
            if name in (
                "lump_sum_day1",
                "monthly_dca_day1",
                "below_sma50_all_in",
                "dip_ytd_10pct",
                "front_loaded_q1",
                "below_sma200_paced",
                "adaptive_dd_pacing",
            ):
                ax.annotate(
                    name,
                    (x.loc[name], y.loc[name]),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=7.5,
                )

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("In-Sample Excess Wealth vs Lump Sum (%)")
        ax.set_ylabel("Out-of-Sample Excess Wealth vs Lump Sum (%)")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Out-of-Sample Generalization Check (Upper-Right Quadrant = Beats Lump Sum in Both)",
        fontsize=12.5,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_deployment_and_cash_yield(
    spy_full: pd.DataFrame,
    spy_rf: pd.DataFrame,
    spy_0_hi: pd.DataFrame,
    spy_rf_hi: pd.DataFrame,
    out_path: Path,
) -> None:
    """Left: Avg deployment day vs excess wealth. Right: ^IRX T-bill cash yield impact."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))

    # Left panel: Time-in-market slope.
    causal = spy_full[spy_full["is_causal"]]
    x = causal["avg_weighted_buy_day"]
    y = causal["diff_vs_lump_pct"] * 100.0
    colors = [CATEGORY_COLORS.get(c, "#333333") for c in causal["category"]]
    axes[0].scatter(x, y, c=colors, s=75, edgecolors="white", linewidth=0.7)
    slope, intercept = np.polyfit(x, y, 1)
    xs = np.linspace(0, float(x.max()) + 5, 100)
    axes[0].plot(
        xs,
        slope * xs + intercept,
        color="#333333",
        linestyle="--",
        linewidth=1.2,
        label=f"Linear fit ({slope * 21:.2f}% / month delayed)",
    )
    for name in (
        "lump_sum_day1",
        "below_sma50_all_in",
        "front_loaded_q1",
        "monthly_dca_day1",
        "dip_ytd_10pct",
        "below_sma200_paced",
    ):
        axes[0].annotate(
            name,
            (x.loc[name], y.loc[name]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7.5,
        )
    axes[0].axhline(0.0, color="#888888", linestyle=":", linewidth=1.0)
    axes[0].set_title("SPY (1994–2025): Deployment Delay vs Excess Wealth", fontweight="bold")
    axes[0].set_xlabel("Value-Weighted Average Buy Day of Year (Trading Days)")
    axes[0].set_ylabel("Terminal Wealth vs Day-1 Lump Sum (%)")
    axes[0].legend(fontsize=8.5)
    axes[0].grid(True, linestyle="--", alpha=0.35)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    # Right panel: 0% cash yield vs ^IRX T-bill yield (full period and high-rate 2022-2025).
    focus = [
        "lump_sum_day1",
        "front_loaded_q1",
        "below_sma50_all_in",
        "hybrid_50_lump_dip10",
        "monthly_dca_day1",
        "adaptive_dd_pacing",
        "dip_ytd_10pct",
        "below_sma200_paced",
    ]
    idx = np.arange(len(focus))
    width = 0.25
    v_full_0 = spy_full.loc[focus, "diff_vs_lump_pct"] * 100.0
    v_full_rf = spy_rf.loc[focus, "diff_vs_lump_pct"] * 100.0
    v_hi_rf = spy_rf_hi.loc[focus, "diff_vs_lump_pct"] * 100.0
    del spy_0_hi

    axes[1].bar(
        idx - width, v_full_0, width=width, label="1994–2025 (0% cash yield)", color="#7f7f7f"
    )
    axes[1].bar(idx, v_full_rf, width=width, label="1994–2025 (^IRX T-bill yield)", color="#1f77b4")
    axes[1].bar(
        idx + width, v_hi_rf, width=width, label="2022–2025 High-Rate (^IRX)", color="#ff7f0e"
    )
    axes[1].axhline(0.0, color="#333333", linewidth=1.0)
    axes[1].set_xticks(idx)
    axes[1].set_xticklabels(focus, rotation=32, ha="right", fontsize=8)
    axes[1].set_ylabel("Terminal Wealth vs Day-1 Lump Sum (%)")
    axes[1].set_title("Impact of Intra-Year Risk-Free Yield (^IRX)", fontweight="bold")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", linestyle="--", alpha=0.35)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_regime_and_rolling(
    spy_regimes: pd.DataFrame,
    spy_rolling_10y: pd.DataFrame,
    out_path: Path,
) -> None:
    """Left: Mean annual share edge vs Lump Sum by market regime. Right: 10y rolling win rates."""
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8))

    focus = [
        "front_loaded_q1",
        "below_sma50_all_in",
        "dip_ytd_5pct",
        "hybrid_50_lump_dip10",
        "monthly_dca_day1",
        "adaptive_dd_pacing",
        "dip_ytd_10pct",
        "below_sma200_paced",
    ]
    piv = spy_regimes.pivot(index="strategy", columns="regime", values="mean_edge_vs_lump_pct")
    piv = piv.loc[focus, ["strong_bull (>=15%)", "moderate_bull (0-15%)", "bear (<0%)"]] * 100.0

    x = np.arange(len(focus))
    w = 0.26
    axes[0].bar(
        x - w, piv["strong_bull (>=15%)"], width=w, label="Strong Bull (>=15%)", color="#2ca02c"
    )
    axes[0].bar(
        x, piv["moderate_bull (0-15%)"], width=w, label="Moderate Bull (0–15%)", color="#1f77b4"
    )
    axes[0].bar(x + w, piv["bear (<0%)"], width=w, label="Bear (<0%)", color="#d62728")
    axes[0].axhline(0.0, color="#333333", linewidth=1.0)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(focus, rotation=32, ha="right", fontsize=8)
    axes[0].set_ylabel("Mean Annual Share Edge vs Lump Sum (%)")
    axes[0].set_title("SPY Regime Breakdown (1994–2025)", fontweight="bold")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", linestyle="--", alpha=0.35)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    # Right: Rolling 10-year win rates vs Lump Sum and Monthly DCA.
    r10 = spy_rolling_10y.sort_values("win_rate_vs_dca", ascending=True)
    y_pos = np.arange(len(r10))
    axes[1].barh(
        y_pos - 0.18,
        r10["win_rate_vs_dca"] * 100.0,
        height=0.35,
        label="Win Rate vs Monthly DCA (%)",
        color="#2ca02c",
        alpha=0.85,
    )
    axes[1].barh(
        y_pos + 0.18,
        r10["win_rate_vs_lump"] * 100.0,
        height=0.35,
        label="Win Rate vs Day-1 Lump Sum (%)",
        color="#1f77b4",
        alpha=0.85,
    )
    axes[1].axvline(50.0, color="#555555", linestyle="--", linewidth=1.0)
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(r10.index, fontsize=8)
    axes[1].set_xlabel("Rolling 10-Year Window Win Rate (%)")
    axes[1].set_title("SPY Rolling 10-Year Win Rates (23 Windows)", fontweight="bold")
    axes[1].set_xlim(0, 108)
    axes[1].legend(loc="lower right", fontsize=8)
    _style_axes(axes[1])

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_synthetic_monte_carlo(
    mc_raw_rel_lump: dict[str, dict[str, np.ndarray]],
    out_path: Path,
) -> None:
    """Boxplots of excess terminal wealth vs Day-1 Lump Sum across N=1,000 synthetic paths."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 6.0), sharey=True)
    order = [
        "front_loaded_q1",
        "below_sma50_all_in",
        "dip_ytd_5pct",
        "hybrid_50_lump_dip10",
        "quarterly_dca",
        "adaptive_dd_pacing",
        "monthly_dca_day1",
        "rsi14_oversold_30",
        "dip_tiered_5_10_15",
        "dip_ytd_10pct",
        "below_sma200_paced",
    ]
    titles = {
        "block_bootstrap": "Stationary Block Bootstrap\n(N=1,000 x 10y, SPY <=2015)",
        "gjr_garch_t": "GJR-GARCH(1,1)-t\n(N=1,000 x 10y, SPY <=2015)",
        "gbm_null": "GBM Null Baseline\n(N=1,000 x 10y, IID Drift)",
    }

    for ax, model in zip(axes, ("block_bootstrap", "gjr_garch_t", "gbm_null"), strict=True):
        data = [mc_raw_rel_lump[model][name] * 100.0 for name in order]
        bp = ax.boxplot(
            data,
            orientation="horizontal",
            showfliers=False,
            patch_artist=True,
            medianprops={"color": "#111111", "linewidth": 1.4},
        )
        for patch in bp["boxes"]:
            patch.set_facecolor("#4c78a8")
            patch.set_alpha(0.65)
        ax.axvline(0.0, color="#d62728", linestyle="--", linewidth=1.2)
        ax.set_title(titles[model], fontsize=10.5, fontweight="bold")
        ax.set_xlabel("Terminal Wealth vs Day-1 Lump Sum (%)")
        _style_axes(ax)

    axes[0].set_yticks(np.arange(1, len(order) + 1))
    axes[0].set_yticklabels(order, fontsize=8.5)
    fig.suptitle(
        "Synthetic Monte Carlo Distribution of Excess Wealth vs Day-1 Lump Sum",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_stress_archetypes(
    stress_tables: dict[str, pd.DataFrame],
    out_path: Path,
) -> None:
    """2x2 comparison of strategy excess wealth vs Day-1 Lump Sum across 4 stress archetypes."""
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.5))
    specs = [
        ("steady_melt_up", "Steady Melt-Up (5 yrs, low-vol rally)"),
        ("v_shaped_flash_crash", "V-Shaped Flash Crash (1 yr, -35% & recovery)"),
        ("slow_bleed_bear", "Slow-Bleed Bear Market (1 yr, -30% grind)"),
        ("lost_decade", "Lost Decade (10 yrs, ~0% net drift, 24% vol)"),
    ]
    focus = [
        "lump_sum_day1",
        "front_loaded_q1",
        "below_sma50_all_in",
        "dip_ytd_5pct",
        "hybrid_50_lump_dip10",
        "monthly_dca_day1",
        "adaptive_dd_pacing",
        "rsi14_oversold_30",
        "dip_ytd_10pct",
        "below_sma200_paced",
    ]

    for ax, (key, title) in zip(axes.flat, specs, strict=True):
        df = stress_tables[key].loc[focus].sort_values("diff_vs_lump_pct", ascending=True)
        vals = df["diff_vs_lump_pct"] * 100.0
        colors = ["#2ca02c" if v > 0 else "#d62728" if v < 0 else "#1f77b4" for v in vals]
        ax.barh(df.index, vals, color=colors, alpha=0.85)
        ax.axvline(0.0, color="#222222", linewidth=1.0)
        ax.set_title(title, fontsize=10.5, fontweight="bold")
        ax.set_xlabel("Terminal Wealth vs Day-1 Lump Sum (%)")
        _style_axes(ax)

    fig.suptitle(
        "Adversarial Stress Archetypes: Excess Wealth vs Day-1 Lump Sum",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def generate_all_figures(study: dict[str, Any], figures_dir: Path = FIGURES_DIR) -> list[Path]:
    """Render all 6 report figures and return their file paths."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        figures_dir / "01_historical_leaderboard.png",
        figures_dir / "02_insample_vs_oos.png",
        figures_dir / "03_deployment_timing_and_cash_yield.png",
        figures_dir / "04_regime_and_rolling_windows.png",
        figures_dir / "05_synthetic_monte_carlo.png",
        figures_dir / "06_stress_archetypes.png",
    ]
    plot_historical_leaderboard(study["spy_full"], study["qqq_full"], paths[0])
    plot_insample_vs_oos(
        study["spy_train"], study["spy_oos"], study["qqq_train"], study["qqq_oos"], paths[1]
    )
    plot_deployment_and_cash_yield(
        study["spy_full"], study["spy_rf"], study["spy_0_hi"], study["spy_rf_hi"], paths[2]
    )
    plot_regime_and_rolling(study["spy_regimes"], study["rolling"]["spy_rolling_10y"], paths[3])
    plot_synthetic_monte_carlo(study["mc_raw_rel_lump"], paths[4])
    plot_stress_archetypes(study["stress_tables"], paths[5])
    return paths


def main() -> None:
    """CLI entrypoint to run the full study and render all figures."""
    study = run_full_study()
    figs = generate_all_figures(study)
    print(f"Generated {len(figs)} figures in {FIGURES_DIR}")


if __name__ == "__main__":
    main()
