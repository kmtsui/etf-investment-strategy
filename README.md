# Annual Index ETF Investment Strategy Analysis

A quantitative data analysis project evaluating buy-only investment strategies under a fixed annual budget constraint for broad-market index ETFs (such as S&P 500 and Nasdaq-100 trackers).

## Problem Statement

Each calendar year (January through December), an investor has a fixed budget $B$ dedicated to purchasing a target index ETF (for example, `SPY` / `VOO` / S&P 500 index or `QQQ` / Nasdaq-100 index).

### Core Constraints

1. **Fixed Annual Budget**: The same budget $B$ is allocated each calendar year (Jan–Dec).
2. **Full Annual Deployment**: 100% of each year's budget must be invested within that calendar year. No cash carries over across calendar years; any unspent cash remaining at year-end is automatically invested on the final trading day of December.
3. **Buy-Only (No Selling)**: Shares can only be purchased and held. Selling, shorting, rebalancing out of equity, or borrowing against future years' budgets is prohibited.
4. **Unrestricted Intra-Year Timing**: Purchases may occur on any trading day(s) within the year in any valid allocation summing to $B$.
5. **Automatic Dividend Reinvestment (DRIP)**: All ETF dividends are assumed to be automatically reinvested into the same ETF (modeled via total-return dividend-adjusted prices or explicit same-day reinvestment).
6. **No Look-Ahead Bias**: All practical strategies must rely strictly on information available on or before the trade execution date.

### Modeling Assumptions (Defaults)

Centralized in `src/etf_strategy/config.py` (`DEFAULT_ASSUMPTIONS`):

- **Target Assets**: `SPY` (S&P 500 ETF) and `QQQ` (Nasdaq-100 ETF), plus `^IRX` (13-week U.S. Treasury Bill yield) for intra-year cash-yield sensitivity.
- **Historical Range**: All complete calendar years (Jan–Dec) available for each ticker; partial inception or in-progress years are excluded.
- **Annual Budget**: \$12,000 per calendar year (scale-invariant under fractional shares; \$1,000/month for Monthly DCA).
- **Execution Price**: Dividend-adjusted close (`adj_close`) on decision day $t$, encoding 100% automatic dividend reinvestment (DRIP).
- **Shares & Fees**: Fractional shares enabled; \$0 commission baseline.
- **Intra-Year Cash Yield**: 0% baseline, plus a sensitivity run earning daily `^IRX` risk-free yield on uninvested intra-year cash.
- **Monthly DCA Benchmark Day**: First trading day of each month (with day-of-month sensitivity across days 1–28 evaluated separately).
- **Chronological Split**: Complete years through 2015 for training / synthetic calibration; 2016 onward held out for out-of-sample evaluation.

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

## Synthetic Data & Stress Testing

Historical ETF records contain only a few dozen calendar years, limiting statistical power and leaving many plausible market paths unobserved. To rigorously test strategies alongside real historical data, the project uses a three-model synthetic suite plus a null-hypothesis baseline:

1. **Stationary Block Bootstrap**: Primary tool for statistical power; resamples contiguous blocks of historical log-returns with random block lengths to preserve volatility clustering, fat tails, and short-term serial dependence across $N = 1,000+$ simulated paths.
2. **GJR-GARCH(1,1) with Student-$t$ Innovations**: Captures unseen tail risk and asymmetric crash volatility (leverage effect), generating drawdowns deeper than historical blocks alone.
3. **Parametric Stress Archetypes**: Adversarial scenarios specifically targeting the 100% annual deployment constraint:
   - *Steady Melt-Up*: Low-volatility upward drift punishing dip-buyers forced into the year-end December sweep.
   - *V-Shaped Flash Crash*: Sharp intra-year drawdown followed by a full recovery.
   - *Slow-Bleed Bear Market*: Persistent year-long decline.
   - *Lost Decade*: Multi-year ~0% net drift with high intra-year volatility.
4. **Geometric Brownian Motion (GBM) Null Baseline**: Used strictly as an engine sanity check rather than a realism model; under positive-drift IID GBM, intra-year timing has zero mathematical edge over Annual Lump Sum (Day 1).

## Repository Structure

```text
.
├── README.md                    # Project overview, problem definition, and methodology
├── AGENTS.md                    # Instructions, strategy interface, and invariants for agents/contributors
├── pyproject.toml               # uv project configuration, dependencies, ruff, and pytest settings
├── .pre-commit-config.yaml      # Local pre-commit hooks (ruff lint/format, pytest)
├── .github/workflows/ci.yml     # GitHub Actions CI for linting and testing
├── data/
│   ├── manifest.json            # Committed data provenance log (ticker, range, fetch time, rows, SHA256)
│   ├── raw/                     # Downloaded historical market data (gitignored)
│   └── processed/               # Cleaned, aligned datasets (gitignored)
├── src/etf_strategy/            # Core data loaders, synthetic generators, backtesting engine, strategies, and metrics
├── notebooks/                   # Exploratory data analysis and visualization notebooks
├── reports/                     # Generated figures, summary tables, and written findings
└── tests/                       # Automated unit tests for engine and strategy invariants
```

## Development Setup

```bash
uv sync --dev
uv run pre-commit install
uv run ruff check .
uv run pytest
```

## Roadmap

- [x] Initialize repository, problem specification (`README.md`), and agent guidelines (`AGENTS.md`)
- [x] Configure `uv` + `pyproject.toml`, `ruff`, `pytest`, `pre-commit`, CI, and `data/manifest.json`
- [x] Finalize modeling assumptions (tickers, date range, execution price, fractional shares, cash yield, fees)
- [x] Acquire and validate historical index/ETF market data (logging provenance in `data/manifest.json`)
- [x] Build the buy-only annual backtesting engine with invariant tests
- [x] Implement and validate synthetic data generators (Stationary Block Bootstrap, GJR-GARCH(1,1)-$t$, Stress Archetypes, GBM baseline) against empirical stylized facts
- [x] Implement benchmarks (Annual Lump Sum, Monthly Fixed-Day DCA, Oracle bounds)
- [x] Implement and backtest candidate strategies
- [ ] Run rolling-window, regime, out-of-sample, synthetic Monte Carlo, and stress-test evaluations
- [ ] Document final conclusions and visual comparisons in `reports/`
