# AGENTS.md

Guidelines, domain invariants, and workflow conventions for AI agents and human contributors working on this repository.

## Project Summary

This project analyzes buy-only annual investment strategies for broad index ETFs (e.g., S&P 500, Nasdaq-100). Each calendar year (January to December), a fixed annual budget must be 100% invested into the target ETF without ever selling shares. Candidate strategies are benchmarked against:

1. **Annual Lump Sum** (investing the full budget on the first trading day of the year).
2. **Monthly Fixed-Day DCA** (investing 1/12 of the annual budget on a fixed day each month).

## Non-Negotiable Domain Invariants

Every backtest, strategy implementation, and analysis script must enforce the following rules:

1. **Zero Look-Ahead Bias**:
   - A strategy's buy decision on trading day $t$ must use only market data available on or before day $t$ (or $t - 1$ if executing at market open).
   - Document the execution price convention explicitly (e.g., signal on close of day $t-1$ executed at open/close of day $t$, or signal and execution at close of day $t$).
   - Hindsight strategies (such as Oracle Best Day / Worst Day) are strictly reference bounds and must be explicitly marked as non-tradable (`is_causal = False`).

2. **Strategy Function Signature**:
   - Every causal strategy must be implemented as a pure function or callable/method with the signature:
     `(price_history_up_to_t, remaining_budget, trading_days_left) -> spend_t`
   - `price_history_up_to_t`: historical price/feature series strictly up to decision point $t$ (never future rows or full-year aggregates).
   - `remaining_budget`: unspent cash remaining in the current calendar year (`>= 0.0`).
   - `trading_days_left`: remaining trading days in the calendar year including day $t$ (`>= 1`).
   - Return value `spend_t`: non-negative float in `[0.0, remaining_budget]`.
   - Keeping every causal strategy pure against this signature makes automated prefix-invariance testing trivial across all strategies.

3. **Exact Annual Budget Exhaustion**:
   - Every complete calendar year must deploy 100% of its annual budget within floating-point tolerance (`1e-6`).
   - Cash may not roll over from one calendar year to the next, and annual spending may never exceed the annual budget.
   - Any unspent cash remaining at the end of a calendar year must be automatically invested on the final trading day of December (`trading_days_left == 1`).

4. **Buy-Only Enforcement**:
   - Daily cash deployed must be non-negative ($\text{spend}_t \ge 0$).
   - Share changes must be non-negative ($\Delta \text{shares}_t \ge 0$).
   - No selling, shorting, leverage, or borrowing against future budgets.

5. **Dividend Reinvestment & Consistent Price Basis**:
   - Assume 100% of dividends are automatically reinvested into the same ETF (DRIP).
   - Compare all strategies on the same total-return price basis (dividend-adjusted close, or explicit same-day dividend reinvestment).
   - If intra-year cash interest (risk-free yield on uninvested budget) or transaction costs are modeled, apply them uniformly and report both base (0% cash yield, \$0 commission) and sensitivity runs.

6. **Overfitting Prevention**:
   - Never optimize hyperparameters across the entire historical dataset and present in-sample results as general superiority.
   - Use chronological train/test splits, walk-forward validation, and parameter sensitivity grids for any parameterized strategy.

## Engineering Conventions

- **Tooling**:
  - Manage Python, dependencies, and lockfile (`uv.lock`) with `uv` and `pyproject.toml`.
  - Lint and format with `ruff`; run unit and invariant tests with `pytest`.
  - Enforce checks locally via `pre-commit` (`.pre-commit-config.yaml`) and in CI via `.github/workflows/ci.yml`.
- **Structure**:
  - `src/etf_strategy/`: Clean, modular, type-annotated Python modules (`data`, `engine`, `strategies`, `metrics`, `plotting`).
  - `tests/`: Automated tests covering engine accounting and strategy invariants.
  - `notebooks/`: Exploratory and presentation notebooks that import logic from `src/etf_strategy/` rather than duplicating backtest code.
  - `reports/`: Reproducible tables, charts, and written analysis outputs.
- **Data Provenance & Auditability**:
  - Keep `data/raw/` and `data/processed/` gitignored.
  - Commit `data/manifest.json` containing metadata for every fetched dataset: `ticker`, `source`, `start_date`, `end_date`, `fetched_at` (UTC ISO-8601), `row_count`, and `sha256` hash of the raw file.
  - Free providers such as `yfinance` silently revise historical `Adj Close` values whenever new dividends are paid; committing `data/manifest.json` keeps `reports/` auditable and surfaces upstream data drift.
- **Reproducibility**:
  - Commit `uv.lock` alongside `pyproject.toml`.
  - Record data manifest hashes, ticker symbols, and exact start/end dates in generated analysis outputs.
  - Fix random seeds for any stochastic or bootstrap procedures.
- **Testing Requirements**:
  - Every causal strategy must pass automated invariant tests verifying:
    - Full annual budget exhaustion for every complete year.
    - Non-negative daily spend (`0.0 <= spend_t <= remaining_budget`) and share deltas.
    - Prefix invariance (calling the strategy with data truncated at $t$ produces the exact same `spend_t` as calling it during a full-history run up to $t$) to mechanically catch look-ahead bugs.

## Git Workflow

- Make small, focused, atomic commits with clear, descriptive messages.
- Do not commit raw downloaded market data, virtual environments, cache directories, or temporary artifacts; keep `.gitignore` up to date.
- Run tests before committing code changes to `src/`.

## Working With the User

- Before writing the backtest engine, confirm or clearly state default assumptions for:
  - Target assets/tickers and historical date range.
  - Execution price convention (Adjusted Close vs. Open/Close).
  - Fractional shares vs. whole shares.
  - Whether uninvested intra-year cash earns a risk-free rate (e.g., T-bill yield) or 0%.
  - Fixed day of month for the Monthly DCA benchmark (e.g., 1st trading day vs. 15th).
- When presenting results, always report exact date ranges, benchmark comparisons, average and median outcomes, win rates, and worst-case regret alongside headline returns.
