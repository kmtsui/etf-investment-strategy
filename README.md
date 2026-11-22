# Annual Index ETF Investment Strategy Analysis

A quantitative data analysis project evaluating buy-only investment strategies under a fixed annual budget constraint for broad-market index ETFs (such as S&P 500 and Nasdaq-100 trackers).

## Problem Statement

Each calendar year (January through December), an investor has a fixed budget $B$ dedicated to purchasing a target index ETF (for example, `SPY` / `VOO` / S&P 500 index or `QQQ` / Nasdaq-100 index).

### Core Constraints

1. **Fixed Annual Budget**: The same budget $B$ is allocated each calendar year (Jan–Dec).
2. **Full Annual Deployment**: 100% of each year's budget must be invested within that calendar year. No cash carries over across calendar years; any unspent cash remaining at year-end is automatically invested on the final trading day of December.
3. **Buy-Only (No Selling)**: Shares can only be purchased and held. Selling, shorting, rebalancing out of equity, or borrowing against future years' budgets is prohibited.
4. **Unrestricted Intra-Year Timing**: Purchases may occur on any trading day(s) within the year in any valid allocation summing to $B$.
5. **No Look-Ahead Bias**: All practical strategies must rely strictly on information available on or before the trade execution date.

## Benchmarks

Every candidate strategy is compared against two primary baselines (plus optional theoretical bounds):

1. **Benchmark Strategy — Annual Lump Sum (Day 1)**: Investing 100% of the annual budget on the first trading day of each year.
2. **Monthly Fixed-Day DCA**: Splitting the annual budget evenly into 12 equal installments and investing on a fixed day each month (e.g., the 1st or 15th calendar day, rolling forward to the next trading day if the market is closed).
3. **Theoretical Bounds (Reference Only)**: Hindsight *Oracle Best Day* (buying 100% at the annual low) and *Worst Day* (buying 100% at the annual high) to bound the maximum possible gain from timing.

## Candidate Strategies

Strategies to design, implement, and test include:

- **Calendar & Frequency Variants**: Weekly, bi-weekly, monthly (sensitivity across days 1–28), and quarterly fixed-schedule DCA.
- **Drawdown / Dip-Buying**: Deploying tranches when price falls $X\%$ below its year-to-date high, trailing peak, or moving average, with a mandatory year-end sweep.
- **Technical / Mean-Reversion Pacing**: Triggering buys on oversold RSI, Bollinger Band lower-band touches, negative multi-day streaks, or price-below-SMA signals.
- **Volatility-Aware Allocation**: Weighting purchase sizes by trailing volatility or VIX regimes.
- **Adaptive Remaining-Budget Pacing**: Dynamically scaling daily or weekly purchases based on current pullback depth and trading days remaining in the year.

## Evaluation Framework

### Primary Metrics

- **Terminal Portfolio Value & Total Shares Accumulated**: Direct comparison of ending wealth under identical annual cash inflows.
- **Average Cost Basis**: Total capital invested divided by total shares acquired.
- **Money-Weighted Return (XIRR / IRR)** and **CAGR**.

### Risk & Robustness Metrics

- **Win Rate vs. Benchmarks**: Percentage of calendar years and rolling multi-year windows (5-year, 10-year, 20-year) outperforming both Annual Lump Sum and Monthly Fixed-Day DCA.
- **Regret / Tail Risk**: Worst-case underperformance relative to benchmarks, maximum drawdown, and volatility.
- **Regime Analysis**: Performance breakdown across bull markets, bear markets, high-volatility periods, and different interest-rate environments.
- **Cash Yield Sensitivity**: Comparing 0% uninvested cash yield versus earning a short-term risk-free rate (e.g., 3-month T-bills) while waiting to deploy intra-year cash.
- **Out-of-Sample Validation**: Chronological train/test splits and walk-forward evaluation to prevent overfitting parameterized strategies.

## Planned Repository Structure

```text
.
├── README.md           # Project overview, problem definition, and methodology
├── AGENTS.md           # Instructions and invariants for AI agents and contributors
├── data/
│   ├── raw/            # Downloaded historical market data (gitignored)
│   └── processed/      # Cleaned, aligned datasets (gitignored)
├── src/                # Core data loaders, backtesting engine, strategies, and metrics
├── notebooks/          # Exploratory data analysis and visualization notebooks
├── reports/            # Generated figures, summary tables, and written findings
└── tests/              # Automated unit tests for engine and strategy invariants
```

## Roadmap

- [x] Initialize repository, problem specification (`README.md`), and agent guidelines (`AGENTS.md`)
- [ ] Finalize modeling assumptions (tickers, date range, execution price, fractional shares, cash yield, fees)
- [ ] Acquire and validate historical index/ETF market data
- [ ] Build the buy-only annual backtesting engine with invariant tests
- [ ] Implement benchmarks (Annual Lump Sum, Monthly Fixed-Day DCA, Oracle bounds)
- [ ] Implement and backtest candidate strategies
- [ ] Run rolling-window, regime, and out-of-sample robustness checks
- [ ] Document final conclusions and visual comparisons in `reports/`
