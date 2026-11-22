"""Core package for annual buy-only index ETF investment strategy analysis."""

from etf_strategy.config import DEFAULT_ASSUMPTIONS, ModelingAssumptions
from etf_strategy.engine import (
    BacktestResult,
    assert_prefix_invariance,
    run_backtest,
    verify_backtest_invariants,
)
from etf_strategy.metrics import compute_xirr, summarize_backtest
from etf_strategy.types import StrategyFn

__all__ = [
    "BacktestResult",
    "DEFAULT_ASSUMPTIONS",
    "ModelingAssumptions",
    "StrategyFn",
    "assert_prefix_invariance",
    "compute_xirr",
    "run_backtest",
    "summarize_backtest",
    "verify_backtest_invariants",
]
