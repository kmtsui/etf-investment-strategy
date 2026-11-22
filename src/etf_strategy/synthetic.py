"""Synthetic market data generators, stress archetypes, and stylized-facts diagnostics.

Implements the three-model core suite plus GBM null baseline specified in README.md:
1. Stationary Block Bootstrap (Politis & Romano, 1994)
2. GJR-GARCH(1,1) with standardized Student-t innovations
3. Parametric Stress Archetypes (Steady Melt-Up, V-Shaped Flash Crash, Slow-Bleed Bear, Lost Decade)
4. Geometric Brownian Motion (GBM) Null Baseline
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GJRGarchParams:
    """Parameters of a daily GJR-GARCH(1,1) model with Student-t innovations."""

    mu: float
    omega: float
    alpha: float
    gamma: float
    beta: float
    nu: float

    @property
    def persistence(self) -> float:
        """Expected variance persistence alpha + 0.5 * gamma + beta."""
        return self.alpha + 0.5 * self.gamma + self.beta

    @property
    def unconditional_variance(self) -> float:
        """Long-run daily return variance omega / (1 - persistence)."""
        denom = max(1.0 - self.persistence, 1e-8)
        return self.omega / denom


def _make_synthetic_price_frame(
    log_returns: np.ndarray,
    *,
    start_year: int = 2001,
    initial_price: float = 100.0,
) -> pd.DataFrame:
    """Build a complete-calendar-year OHLCV DataFrame from daily log-returns."""
    n = len(log_returns)
    # Generate enough business days and map exact length.
    n_years_est = max(1, math.ceil(n / 250) + 2)
    dates = pd.bdate_range(start=f"{start_year}-01-01", periods=n)
    if len(dates) != n:
        dates = pd.bdate_range(
            start=f"{start_year}-01-01", end=f"{start_year + n_years_est}-12-31"
        )[:n]

    cum_log_ret = np.cumsum(log_returns)
    close = initial_price * np.exp(cum_log_ret)
    prev_close = np.empty_like(close)
    prev_close[0] = initial_price
    prev_close[1:] = close[:-1]

    high = np.maximum(prev_close, close)
    low = np.minimum(prev_close, close)

    return pd.DataFrame(
        {
            "open": prev_close,
            "high": high,
            "low": low,
            "close": close,
            "adj_close": close,
            "volume": np.full(n, 1_000_000.0, dtype=float),
        },
        index=pd.DatetimeIndex(dates, name="date"),
    )


def _calendar_dates_for_years(n_years: int, start_year: int = 2001) -> pd.DatetimeIndex:
    """Return business-day DatetimeIndex covering exactly `n_years` complete calendar years."""
    if n_years < 1:
        raise ValueError("n_years must be >= 1.")
    return pd.bdate_range(start=f"{start_year}-01-01", end=f"{start_year + n_years - 1}-12-31")


def _returns_to_complete_year_frame(
    log_returns: np.ndarray,
    dates: pd.DatetimeIndex,
    *,
    initial_price: float = 100.0,
) -> pd.DataFrame:
    """Convert daily log-returns aligned to `dates` into an OHLCV DataFrame."""
    if len(log_returns) != len(dates):
        raise ValueError(f"Expected {len(dates)} returns, got {len(log_returns)}.")
    close = initial_price * np.exp(np.cumsum(log_returns))
    prev_close = np.empty_like(close)
    prev_close[0] = initial_price
    prev_close[1:] = close[:-1]
    return pd.DataFrame(
        {
            "open": prev_close,
            "high": np.maximum(prev_close, close),
            "low": np.minimum(prev_close, close),
            "close": close,
            "adj_close": close,
            "volume": np.full(len(dates), 1_000_000.0, dtype=float),
        },
        index=pd.DatetimeIndex(dates, name="date"),
    )


def extract_log_returns(
    price_df: pd.DataFrame,
    *,
    price_col: str = "adj_close",
    train_end_year: int | None = None,
) -> np.ndarray:
    """Extract daily log-returns, optionally restricting strictly to `year <= train_end_year`."""
    df = price_df
    if train_end_year is not None:
        df = df[df.index.year <= train_end_year]
    prices = df[price_col].dropna().to_numpy(dtype=float)
    if len(prices) < 30:
        raise ValueError("Insufficient price history to extract log-returns (need >= 30 rows).")
    if np.any(prices <= 0.0):
        raise ValueError("Prices must be strictly positive to compute log-returns.")
    return np.diff(np.log(prices))


# ---------------------------------------------------------------------------
# 1. Stationary Block Bootstrap (Politis & Romano, 1994)
# ---------------------------------------------------------------------------


def generate_stationary_block_bootstrap(
    price_df: pd.DataFrame,
    *,
    n_years: int,
    n_paths: int = 1,
    expected_block_length: float = 20.0,
    seed: int,
    train_end_year: int | None = None,
    price_col: str = "adj_close",
    start_year: int = 2001,
    initial_price: float = 100.0,
) -> list[pd.DataFrame]:
    """Generate synthetic price paths via Stationary Block Bootstrap.

    Draws contiguous circular blocks of historical log-returns whose lengths follow a
    geometric distribution with mean `expected_block_length`, preserving volatility
    clustering, fat tails, and short-term serial dependence.
    """
    if expected_block_length < 1.0:
        raise ValueError("expected_block_length must be >= 1.0.")
    if n_paths < 1:
        raise ValueError("n_paths must be >= 1.")

    hist_returns = extract_log_returns(
        price_df,
        price_col=price_col,
        train_end_year=train_end_year,
    )
    n_hist = len(hist_returns)
    dates = _calendar_dates_for_years(n_years, start_year=start_year)
    n_steps = len(dates)

    p_new_block = 1.0 / expected_block_length
    rng = np.random.default_rng(seed)

    paths: list[pd.DataFrame] = []
    for _ in range(n_paths):
        idx = np.empty(n_steps, dtype=np.int64)
        idx[0] = int(rng.integers(0, n_hist))
        switches = rng.random(n_steps) < p_new_block
        starts = rng.integers(0, n_hist, size=n_steps)
        for t in range(1, n_steps):
            if switches[t]:
                idx[t] = starts[t]
            else:
                idx[t] = (idx[t - 1] + 1) % n_hist
        sampled_returns = hist_returns[idx]
        paths.append(
            _returns_to_complete_year_frame(sampled_returns, dates, initial_price=initial_price)
        )
    return paths


# ---------------------------------------------------------------------------
# 2. GJR-GARCH(1,1) with Student-t Innovations
# ---------------------------------------------------------------------------


def _gjr_garch_neg_loglik(theta: np.ndarray, eps: np.ndarray, sample_var: float) -> float:
    """Negative log-likelihood of GJR-GARCH(1,1)-t on demeaned returns `eps`."""
    # Map unconstrained reals -> valid stationary GJR-GARCH(1,1)-t parameters.
    omega = math.exp(float(np.clip(theta[0], -30.0, 5.0)))
    # Softmax-like mapping so alpha >= 0, beta >= 0, gamma >= 0, and alpha + 0.5*gamma + beta < 1.
    u_a = math.exp(float(np.clip(theta[1], -20.0, 20.0)))
    u_g = math.exp(float(np.clip(theta[2], -20.0, 20.0)))
    u_b = math.exp(float(np.clip(theta[3], -20.0, 20.0)))
    denom = 1.0 + u_a + 0.5 * u_g + u_b
    alpha = u_a / denom
    gamma = u_g / denom
    beta = u_b / denom
    nu = 2.1 + math.exp(float(np.clip(theta[4], -10.0, 5.0)))

    eps2 = eps * eps
    neg = (eps < 0.0).astype(float)
    n = len(eps)

    h = np.empty(n, dtype=float)
    h[0] = sample_var
    for t in range(1, n):
        h[t] = omega + (alpha + gamma * neg[t - 1]) * eps2[t - 1] + beta * h[t - 1]

    if np.any(h <= 1e-18) or not np.all(np.isfinite(h)):
        return 1e12

    const = (
        math.lgamma(0.5 * (nu + 1.0)) - math.lgamma(0.5 * nu) - 0.5 * math.log(math.pi * (nu - 2.0))
    )
    ll = np.sum(const - 0.5 * np.log(h) - 0.5 * (nu + 1.0) * np.log1p(eps2 / ((nu - 2.0) * h)))
    if not math.isfinite(ll):
        return 1e12
    return -float(ll)


def _unpack_gjr_params(theta: np.ndarray, mu: float) -> GJRGarchParams:
    omega = math.exp(float(np.clip(theta[0], -30.0, 5.0)))
    u_a = math.exp(float(np.clip(theta[1], -20.0, 20.0)))
    u_g = math.exp(float(np.clip(theta[2], -20.0, 20.0)))
    u_b = math.exp(float(np.clip(theta[3], -20.0, 20.0)))
    denom = 1.0 + u_a + 0.5 * u_g + u_b
    return GJRGarchParams(
        mu=mu,
        omega=omega,
        alpha=u_a / denom,
        gamma=u_g / denom,
        beta=u_b / denom,
        nu=2.1 + math.exp(float(np.clip(theta[4], -10.0, 5.0))),
    )


def _nelder_mead(
    func: Any,
    x0: np.ndarray,
    *,
    step: np.ndarray,
    max_iter: int = 350,
    tol: float = 1e-6,
) -> tuple[np.ndarray, float]:
    """Lightweight deterministic Nelder-Mead optimizer in pure NumPy."""
    n = len(x0)
    simplex = np.tile(x0, (n + 1, 1)).astype(float)
    for i in range(n):
        simplex[i + 1, i] += step[i]

    f_vals = np.array([func(v) for v in simplex], dtype=float)

    for _ in range(max_iter):
        order = np.argsort(f_vals)
        simplex = simplex[order]
        f_vals = f_vals[order]

        if f_vals[-1] - f_vals[0] < tol:
            break

        centroid = simplex[:-1].mean(axis=0)
        reflected = centroid + (centroid - simplex[-1])
        f_ref = func(reflected)

        if f_vals[0] <= f_ref < f_vals[-2]:
            simplex[-1] = reflected
            f_vals[-1] = f_ref
        elif f_ref < f_vals[0]:
            expanded = centroid + 2.0 * (reflected - centroid)
            f_exp = func(expanded)
            if f_exp < f_ref:
                simplex[-1] = expanded
                f_vals[-1] = f_exp
            else:
                simplex[-1] = reflected
                f_vals[-1] = f_ref
        else:
            contracted = centroid + 0.5 * (simplex[-1] - centroid)
            f_con = func(contracted)
            if f_con < f_vals[-1]:
                simplex[-1] = contracted
                f_vals[-1] = f_con
            else:
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    f_vals[i] = func(simplex[i])

    best_idx = int(np.argmin(f_vals))
    return simplex[best_idx], float(f_vals[best_idx])


def fit_gjr_garch_t(
    price_df: pd.DataFrame,
    *,
    train_end_year: int | None = None,
    price_col: str = "adj_close",
) -> GJRGarchParams:
    """Fit GJR-GARCH(1,1) with Student-t innovations via MLE strictly on the training data."""
    returns = extract_log_returns(
        price_df,
        price_col=price_col,
        train_end_year=train_end_year,
    )
    sample_mean = float(np.mean(returns))
    sample_var = float(np.var(returns, ddof=1))

    # Mean-targeting: anchor mu to the empirical training sample mean so Student-t MLE
    # downweighting of crash outliers does not artificially inflate simulated drift.
    eps = returns - sample_mean
    init_omega = max(0.02 * sample_var, 1e-8)
    x0 = np.array(
        [
            math.log(init_omega),
            math.log(0.5),
            math.log(3.0),
            math.log(22.0),
            math.log(5.0),
        ],
        dtype=float,
    )
    step = np.array([0.5, 0.5, 0.5, 0.5, 0.5], dtype=float)
    best_theta, _ = _nelder_mead(
        lambda th: _gjr_garch_neg_loglik(th, eps, sample_var),
        x0,
        step=step,
        max_iter=350,
    )
    return _unpack_gjr_params(best_theta, mu=sample_mean)


def generate_gjr_garch_t(
    params: GJRGarchParams,
    *,
    n_years: int,
    n_paths: int = 1,
    seed: int,
    burn_in: int = 250,
    start_year: int = 2001,
    initial_price: float = 100.0,
) -> list[pd.DataFrame]:
    """Simulate synthetic price paths from a calibrated GJR-GARCH(1,1)-t model."""
    if n_paths < 1:
        raise ValueError("n_paths must be >= 1.")
    if params.nu <= 2.0:
        raise ValueError("Student-t degrees of freedom nu must be > 2.0.")

    dates = _calendar_dates_for_years(n_years, start_year=start_year)
    n_steps = len(dates)
    total_steps = n_steps + burn_in

    rng = np.random.default_rng(seed)
    t_scale = math.sqrt((params.nu - 2.0) / params.nu)
    h0 = params.unconditional_variance

    paths: list[pd.DataFrame] = []
    for _ in range(n_paths):
        z = rng.standard_t(df=params.nu, size=total_steps) * t_scale
        returns = np.empty(total_steps, dtype=float)
        h = h0
        eps = 0.0
        for t in range(total_steps):
            if t > 0:
                indicator = 1.0 if eps < 0.0 else 0.0
                h = (
                    params.omega
                    + (params.alpha + params.gamma * indicator) * (eps * eps)
                    + (params.beta * h)
                )
            eps = math.sqrt(max(h, 1e-18)) * z[t]
            returns[t] = params.mu + eps

        sim_returns = returns[burn_in:]
        paths.append(
            _returns_to_complete_year_frame(sim_returns, dates, initial_price=initial_price)
        )
    return paths


# ---------------------------------------------------------------------------
# 3. Parametric Stress Archetypes
# ---------------------------------------------------------------------------


def generate_stress_archetype(
    archetype: str,
    *,
    seed: int,
    n_years: int | None = None,
    start_year: int = 2001,
    initial_price: float = 100.0,
) -> pd.DataFrame:
    """Generate one of the four adversarial stress scenarios targeting the annual constraint.

    Supported archetypes:
    - 'steady_melt_up': Low-vol upward drift punishing dip-buyers forced into the Dec sweep.
    - 'v_shaped_flash_crash': Sharp intra-year drawdown followed by full recovery.
    - 'slow_bleed_bear': Persistent year-long decline.
    - 'lost_decade': Multi-year ~0% net drift with high intra-year volatility.
    """
    rng = np.random.default_rng(seed)

    if archetype == "steady_melt_up":
        years = 5 if n_years is None else n_years
        dates = _calendar_dates_for_years(years, start_year=start_year)
        n = len(dates)
        mu_daily = 0.18 / 252.0
        sigma_daily = 0.06 / math.sqrt(252.0)
        log_ret = mu_daily - 0.5 * sigma_daily**2 + sigma_daily * rng.standard_normal(n)
        return _returns_to_complete_year_frame(log_ret, dates, initial_price=initial_price)

    if archetype == "v_shaped_flash_crash":
        years = 1 if n_years is None else n_years
        dates = _calendar_dates_for_years(years, start_year=start_year)
        log_ret = np.zeros(len(dates), dtype=float)
        noise_sigma = 0.08 / math.sqrt(252.0)
        for year in range(start_year, start_year + years):
            mask = np.flatnonzero(dates.year == year)
            m = len(mask)
            t_peak = int(0.15 * m)
            t_trough = int(0.30 * m)
            t_recov = int(0.65 * m)

            # Rally +5%, crash -35%, recover back to +10% YTD.
            yr_ret = np.zeros(m, dtype=float)
            yr_ret[:t_peak] = math.log(1.05) / max(t_peak, 1)
            yr_ret[t_peak:t_trough] = math.log(0.65) / max(t_trough - t_peak, 1)
            yr_ret[t_trough:t_recov] = math.log(1.05 / 0.65) / max(t_recov - t_trough, 1)
            yr_ret[t_recov:] = math.log(1.08 / 1.05) / max(m - t_recov, 1)
            yr_ret += rng.normal(0.0, noise_sigma, size=m)
            log_ret[mask] = yr_ret
        return _returns_to_complete_year_frame(log_ret, dates, initial_price=initial_price)

    if archetype == "slow_bleed_bear":
        years = 1 if n_years is None else n_years
        dates = _calendar_dates_for_years(years, start_year=start_year)
        n = len(dates)
        # Persistent ~-30%/yr decline with realistic bear-market volatility.
        mu_daily = math.log(0.70) / 252.0
        sigma_daily = 0.16 / math.sqrt(252.0)
        noise = rng.normal(0.0, sigma_daily, size=n)
        # Demean noise so the realized path preserves the slow-bleed bear drift cleanly.
        log_ret = mu_daily + (noise - noise.mean())
        return _returns_to_complete_year_frame(log_ret, dates, initial_price=initial_price)

    if archetype == "lost_decade":
        years = 10 if n_years is None else n_years
        dates = _calendar_dates_for_years(years, start_year=start_year)
        n = len(dates)
        # High-volatility sideways market with ~0% net 10-year log return.
        sigma_daily = 0.24 / math.sqrt(252.0)
        noise = rng.normal(0.0, sigma_daily, size=n)
        # Add cyclical multi-month swings plus exact zero-mean adjustment -> ~0% net drift.
        t = np.arange(n, dtype=float)
        cycle = 0.0008 * np.sin(2.0 * math.pi * t / 300.0)
        raw = noise + cycle
        log_ret = raw - raw.mean()
        return _returns_to_complete_year_frame(log_ret, dates, initial_price=initial_price)

    supported = ("steady_melt_up", "v_shaped_flash_crash", "slow_bleed_bear", "lost_decade")
    raise ValueError(f"Unknown stress archetype '{archetype}'. Expected one of {supported}.")


# ---------------------------------------------------------------------------
# 4. Geometric Brownian Motion (GBM) Null Baseline
# ---------------------------------------------------------------------------


def generate_gbm_paths(
    *,
    n_years: int,
    n_paths: int = 1,
    annual_drift: float = 0.09,
    annual_vol: float = 0.16,
    seed: int,
    start_year: int = 2001,
    initial_price: float = 100.0,
) -> list[pd.DataFrame]:
    """Generate positive-drift IID Geometric Brownian Motion paths for null-hypothesis tests."""
    if n_paths < 1:
        raise ValueError("n_paths must be >= 1.")
    if annual_vol < 0.0:
        raise ValueError("annual_vol must be >= 0.0.")

    dates = _calendar_dates_for_years(n_years, start_year=start_year)
    n_steps = len(dates)
    dt = 1.0 / 252.0
    drift_step = (annual_drift - 0.5 * annual_vol**2) * dt
    vol_step = annual_vol * math.sqrt(dt)

    rng = np.random.default_rng(seed)
    paths: list[pd.DataFrame] = []
    for _ in range(n_paths):
        log_ret = drift_step + vol_step * rng.standard_normal(n_steps)
        paths.append(_returns_to_complete_year_frame(log_ret, dates, initial_price=initial_price))
    return paths


# ---------------------------------------------------------------------------
# Stylized-Facts Diagnostics
# ---------------------------------------------------------------------------


def _autocorr(x: np.ndarray, lag: int) -> float:
    if len(x) <= lag + 1:
        return float("nan")
    a = x[:-lag]
    b = x[lag:]
    if np.std(a) == 0.0 or np.std(b) == 0.0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def compute_stylized_facts(
    price_df: pd.DataFrame,
    *,
    price_col: str = "adj_close",
) -> dict[str, float]:
    """Compute empirical stylized-facts diagnostics for a price series.

    Diagnostics match AGENTS.md Synthetic Data Integrity requirements:
    - annualized_drift (log-return drift) and annualized_vol
    - skewness and excess_kurtosis
    - intra-year max drawdown quantiles (10th, 50th, 90th percentiles)
    - autocorrelation of squared returns at lags 1, 5, and 20
    - leverage correlation corr(r_t, r_{t+1}^2)
    """
    returns = extract_log_returns(price_df, price_col=price_col)
    mean_r = float(np.mean(returns))
    std_r = float(np.std(returns, ddof=1))
    standardized = (returns - mean_r) / std_r if std_r > 0.0 else np.zeros_like(returns)

    skewness = float(np.mean(standardized**3))
    excess_kurtosis = float(np.mean(standardized**4) - 3.0)

    sq = returns * returns
    acf_sq_1 = _autocorr(sq, 1)
    acf_sq_5 = _autocorr(sq, 5)
    acf_sq_20 = _autocorr(sq, 20)
    leverage_corr = (
        float(np.corrcoef(returns[:-1], sq[1:])[0, 1]) if len(returns) > 2 and std_r > 0.0 else 0.0
    )

    intra_year_dds: list[float] = []
    for _, group in price_df.groupby(price_df.index.year):
        if len(group) < 20:
            continue
        p = group[price_col].to_numpy(dtype=float)
        running_peak = np.maximum.accumulate(p)
        dd = float(np.min(p / running_peak - 1.0))
        intra_year_dds.append(dd)

    dd_arr = np.array(intra_year_dds, dtype=float)
    q10, q50, q90 = (
        np.quantile(dd_arr, [0.10, 0.50, 0.90]) if len(dd_arr) > 0 else (np.nan, np.nan, np.nan)
    )

    return {
        "annualized_drift": mean_r * 252.0,
        "annualized_vol": std_r * math.sqrt(252.0),
        "skewness": skewness,
        "excess_kurtosis": excess_kurtosis,
        "intra_year_max_dd_p10": float(q10),
        "intra_year_max_dd_p50": float(q50),
        "intra_year_max_dd_p90": float(q90),
        "sq_return_acf_lag1": acf_sq_1,
        "sq_return_acf_lag5": acf_sq_5,
        "sq_return_acf_lag20": acf_sq_20,
        "leverage_corr": leverage_corr,
    }


def summarize_synthetic_stylized_facts(
    paths: list[pd.DataFrame],
    *,
    price_col: str = "adj_close",
) -> pd.DataFrame:
    """Aggregate stylized-facts diagnostics (mean, p10, p50, p90) across simulated paths."""
    records = [compute_stylized_facts(df, price_col=price_col) for df in paths]
    frame = pd.DataFrame(records)
    return pd.DataFrame(
        {
            "mean": frame.mean(),
            "p10": frame.quantile(0.10),
            "p50": frame.quantile(0.50),
            "p90": frame.quantile(0.90),
        }
    )
