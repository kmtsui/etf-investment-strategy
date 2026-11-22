"""Core package for annual buy-only index ETF investment strategy analysis."""

from etf_strategy.config import DEFAULT_ASSUMPTIONS, ModelingAssumptions
from etf_strategy.engine import (
    BacktestResult,
    assert_prefix_invariance,
    run_backtest,
    verify_backtest_invariants,
)
from etf_strategy.metrics import compute_xirr, summarize_backtest
from etf_strategy.synthetic import (
    GJRGarchParams,
    compute_stylized_facts,
    fit_gjr_garch_t,
    generate_gbm_paths,
    generate_gjr_garch_t,
    generate_stationary_block_bootstrap,
    generate_stress_archetype,
    summarize_synthetic_stylized_facts,
)
from etf_strategy.types import StrategyFn

__all__ = [
    "BacktestResult",
    "DEFAULT_ASSUMPTIONS",
    "GJRGarchParams",
    "ModelingAssumptions",
    "StrategyFn",
    "assert_prefix_invariance",
    "compute_stylized_facts",
    "compute_xirr",
    "fit_gjr_garch_t",
    "generate_gbm_paths",
    "generate_gjr_garch_t",
    "generate_stationary_block_bootstrap",
    "generate_stress_archetype",
    "run_backtest",
    "summarize_backtest",
    "summarize_synthetic_stylized_facts",
    "verify_backtest_invariants",
]
