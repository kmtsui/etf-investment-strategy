"""Centralized modeling assumptions for the annual ETF strategy study."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelingAssumptions:
    """Explicit, reproducible defaults for historical and synthetic backtests."""

    tickers: tuple[str, ...] = ("SPY", "QQQ")
    risk_free_ticker: str = "^IRX"
    annual_budget: float = 12_000.0
    execution_price_col: str = "adj_close"
    fractional_shares: bool = True
    commission_per_trade: float = 0.0
    baseline_cash_yield: float = 0.0
    monthly_dca_day: int = 1
    budget_tolerance: float = 1e-6
    complete_years_only: bool = True
    train_end_year: int = 2015


DEFAULT_ASSUMPTIONS = ModelingAssumptions()
