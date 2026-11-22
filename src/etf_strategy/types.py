"""Shared type definitions for buy-only annual ETF investment strategies."""

from typing import Protocol

import pandas as pd


class StrategyFn(Protocol):
    """Pure causal strategy interface returning cash to spend on trading day t."""

    def __call__(
        self,
        price_history_up_to_t: pd.DataFrame,
        remaining_budget: float,
        trading_days_left: int,
    ) -> float: ...
