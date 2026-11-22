"""Tests for synthetic generators, stylized facts, and the GBM null-hypothesis sanity check."""

import numpy as np
import pandas as pd
import pytest

from etf_strategy.engine import run_backtest, verify_backtest_invariants
from etf_strategy.metrics import summarize_backtest
from etf_strategy.synthetic import (
    GJRGarchParams,
    compute_stylized_facts,
    extract_log_returns,
    fit_gjr_garch_t,
    generate_gbm_paths,
    generate_gjr_garch_t,
    generate_stationary_block_bootstrap,
    generate_stress_archetype,
    summarize_synthetic_stylized_facts,
)


def _lump_sum(hist: pd.DataFrame, rem: float, left: int) -> float:
    del hist, left
    return rem


def _equal_daily_dca(hist: pd.DataFrame, rem: float, left: int) -> float:
    del hist
    return rem / left


def _dip_buyer_5pct(hist: pd.DataFrame, rem: float, left: int) -> float:
    prices = hist["adj_close"].to_numpy()
    window = prices[-min(len(prices), 60) :]
    drawdown = float(prices[-1] / window.max()) - 1.0
    if drawdown <= -0.05:
        return rem * 0.5
    return 0.0 if left > 1 else rem


def _make_clustered_vol_history(seed: int = 7) -> pd.DataFrame:
    """Create a realistic 15-year GJR-GARCH-t history for offline bootstrap/fit tests."""
    params = GJRGarchParams(
        mu=0.00035,
        omega=2.0e-6,
        alpha=0.02,
        gamma=0.12,
        beta=0.89,
        nu=6.0,
    )
    return generate_gjr_garch_t(params, n_years=15, n_paths=1, seed=seed, start_year=2005)[0]


def test_deterministic_seeding_across_all_generators() -> None:
    hist = _make_clustered_vol_history(seed=11)

    b1 = generate_stationary_block_bootstrap(hist, n_years=3, n_paths=2, seed=42)
    b2 = generate_stationary_block_bootstrap(hist, n_years=3, n_paths=2, seed=42)
    b3 = generate_stationary_block_bootstrap(hist, n_years=3, n_paths=2, seed=43)
    pd.testing.assert_frame_equal(b1[0], b2[0])
    assert not b1[0]["adj_close"].equals(b3[0]["adj_close"])

    params = GJRGarchParams(mu=0.0003, omega=2e-6, alpha=0.03, gamma=0.10, beta=0.88, nu=6.5)
    g1 = generate_gjr_garch_t(params, n_years=3, n_paths=1, seed=99)[0]
    g2 = generate_gjr_garch_t(params, n_years=3, n_paths=1, seed=99)[0]
    pd.testing.assert_frame_equal(g1, g2)

    m1 = generate_gbm_paths(n_years=3, n_paths=1, seed=123)[0]
    m2 = generate_gbm_paths(n_years=3, n_paths=1, seed=123)[0]
    pd.testing.assert_frame_equal(m1, m2)

    s1 = generate_stress_archetype("lost_decade", seed=55)
    s2 = generate_stress_archetype("lost_decade", seed=55)
    pd.testing.assert_frame_equal(s1, s2)


def test_no_calibration_leakage_respects_train_end_year() -> None:
    hist = _make_clustered_vol_history(seed=19)
    # Shock the held-out years (2016+) dramatically; training extraction must be unaffected.
    shocked = hist.copy()
    shocked.loc[shocked.index.year > 2015, "adj_close"] *= 50.0

    r_clean = extract_log_returns(hist, train_end_year=2015)
    r_shocked = extract_log_returns(shocked, train_end_year=2015)
    np.testing.assert_allclose(r_clean, r_shocked)

    b_clean = generate_stationary_block_bootstrap(
        hist, n_years=2, n_paths=1, seed=7, train_end_year=2015
    )[0]
    b_shocked = generate_stationary_block_bootstrap(
        shocked, n_years=2, n_paths=1, seed=7, train_end_year=2015
    )[0]
    pd.testing.assert_frame_equal(b_clean, b_shocked)


def test_gjr_garch_fit_and_stylized_facts_validation() -> None:
    hist = _make_clustered_vol_history(seed=23)
    fitted = fit_gjr_garch_t(hist, train_end_year=2015)

    assert fitted.omega > 0.0
    assert fitted.alpha >= 0.0
    assert fitted.gamma > 0.0  # Captures asymmetric crash volatility (leverage effect)
    assert fitted.beta > 0.5
    assert fitted.persistence < 1.0
    assert 2.1 < fitted.nu < 30.0

    garch_paths = generate_gjr_garch_t(fitted, n_years=10, n_paths=15, seed=101)
    summary = summarize_synthetic_stylized_facts(garch_paths)
    # Fat tails and positive volatility clustering must be present.
    assert float(summary.loc["excess_kurtosis", "mean"]) > 0.5
    assert float(summary.loc["sq_return_acf_lag1", "mean"]) > 0.05

    boot_paths = generate_stationary_block_bootstrap(
        hist, n_years=10, n_paths=15, expected_block_length=20.0, seed=102, train_end_year=2015
    )
    boot_summary = summarize_synthetic_stylized_facts(boot_paths)
    assert float(boot_summary.loc["excess_kurtosis", "mean"]) > 0.5
    assert float(boot_summary.loc["sq_return_acf_lag1", "mean"]) > 0.05


def test_parametric_stress_archetypes_behave_as_specified() -> None:
    melt = generate_stress_archetype("steady_melt_up", seed=1, n_years=3)
    melt_facts = compute_stylized_facts(melt)
    assert melt_facts["annualized_drift"] > 0.10
    assert melt_facts["annualized_vol"] < 0.10
    assert melt_facts["intra_year_max_dd_p50"] > -0.10

    vcrash = generate_stress_archetype("v_shaped_flash_crash", seed=2, n_years=1)
    vcrash_facts = compute_stylized_facts(vcrash)
    assert vcrash_facts["intra_year_max_dd_p50"] < -0.25
    assert float(vcrash["adj_close"].iloc[-1]) > float(vcrash["adj_close"].iloc[0])

    bear = generate_stress_archetype("slow_bleed_bear", seed=3, n_years=1)
    assert float(bear["adj_close"].iloc[-1]) < 0.80 * float(bear["adj_close"].iloc[0])

    lost = generate_stress_archetype("lost_decade", seed=4, n_years=10)
    lost_facts = compute_stylized_facts(lost)
    assert abs(lost_facts["annualized_drift"]) < 0.01
    assert lost_facts["annualized_vol"] > 0.18

    # All four archetypes must pass backtest invariants cleanly.
    for name, df in (
        ("melt", melt),
        ("vcrash", vcrash),
        ("bear", bear),
        ("lost", lost),
    ):
        res = run_backtest(df, _dip_buyer_5pct, strategy_name=f"dip_{name}")
        verify_backtest_invariants(res)

    with pytest.raises(ValueError, match="Unknown stress archetype"):
        generate_stress_archetype("invalid_regime", seed=1)


def test_gbm_null_hypothesis_sanity() -> None:
    """Under positive-drift IID GBM, no causal timing strategy beats Day-1 Lump Sum in expectation.

    Mechanically catches subtle look-ahead leaks or engine accounting bugs (AGENTS.md).
    """
    paths = generate_gbm_paths(
        n_years=3,
        n_paths=60,
        annual_drift=0.10,
        annual_vol=0.16,
        seed=2026,
    )

    lump_vals: list[float] = []
    dca_vals: list[float] = []
    dip_vals: list[float] = []

    for df in paths:
        lump_vals.append(summarize_backtest(run_backtest(df, _lump_sum))["terminal_value"])
        dca_vals.append(summarize_backtest(run_backtest(df, _equal_daily_dca))["terminal_value"])
        dip_vals.append(summarize_backtest(run_backtest(df, _dip_buyer_5pct))["terminal_value"])

    mean_lump = float(np.mean(lump_vals))
    mean_dca = float(np.mean(dca_vals))
    mean_dip = float(np.mean(dip_vals))

    assert mean_lump > mean_dca
    assert mean_lump > mean_dip

    # IID GBM must also have near-zero excess kurtosis and near-zero squared-return autocorr.
    gbm_summary = summarize_synthetic_stylized_facts(paths[:20])
    assert abs(float(gbm_summary.loc["excess_kurtosis", "mean"])) < 0.5
    assert abs(float(gbm_summary.loc["sq_return_acf_lag1", "mean"])) < 0.05
