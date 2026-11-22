"""Historical market data fetching, validation, and manifest provenance tracking."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf

from etf_strategy.config import DEFAULT_ASSUMPTIONS

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = REPO_ROOT / "data" / "raw"
DEFAULT_MANIFEST_PATH = REPO_ROOT / "data" / "manifest.json"

REQUIRED_ETF_COLUMNS = ("open", "high", "low", "close", "adj_close", "volume")
REQUIRED_YIELD_COLUMNS = ("close",)


def safe_ticker_filename(ticker: str) -> str:
    """Convert a ticker symbol such as '^IRX' into a filesystem-safe CSV filename."""
    cleaned = ticker.strip().lstrip("^").replace("/", "_").replace(".", "_")
    return f"{cleaned}.csv"


def compute_sha256(path: Path) -> str:
    """Return the lowercase hexadecimal SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65_536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_history_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize a yfinance history DataFrame into snake_case columns and tz-naive dates."""
    if raw.empty:
        raise ValueError("Received empty market data frame.")

    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    rename_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
        "Dividends": "dividends",
        "Stock Splits": "stock_splits",
        "Capital Gains": "capital_gains",
    }
    df = df.rename(columns={col: rename_map.get(str(col), str(col).lower()) for col in df.columns})

    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize()
    df.index.name = "date"
    df = df[~df.index.duplicated(keep="last")].sort_index()

    if "adj_close" not in df.columns and "close" in df.columns:
        df["adj_close"] = df["close"]
    if "dividends" not in df.columns:
        df["dividends"] = 0.0
    if "stock_splits" not in df.columns:
        df["stock_splits"] = 0.0

    preferred = [
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
        "dividends",
        "stock_splits",
    ]
    ordered = [col for col in preferred if col in df.columns]
    return df[ordered]


def complete_calendar_years(df: pd.DataFrame, min_trading_days: int = 240) -> list[int]:
    """Return sorted calendar years with full Jan-Dec trading coverage."""
    if df.empty:
        return []

    years: list[int] = []
    for year, group in df.groupby(df.index.year):
        first_date = group.index.min()
        last_date = group.index.max()
        starts_in_early_jan = first_date.month == 1 and first_date.day <= 10
        ends_in_late_dec = last_date.month == 12 and last_date.day >= 20
        if starts_in_early_jan and ends_in_late_dec and len(group) >= min_trading_days:
            years.append(int(year))
    return sorted(years)


def filter_complete_years(df: pd.DataFrame, min_trading_days: int = 240) -> pd.DataFrame:
    """Filter a price DataFrame to complete calendar years only."""
    valid_years = set(complete_calendar_years(df, min_trading_days=min_trading_days))
    if not valid_years:
        return df.iloc[0:0].copy()
    return df[df.index.year.isin(valid_years)].copy()


def validate_price_frame(
    df: pd.DataFrame,
    ticker: str,
    *,
    is_yield: bool = False,
) -> dict[str, Any]:
    """Validate historical data integrity and return summary diagnostics."""
    if df.empty:
        raise ValueError(f"{ticker}: dataset is empty.")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"{ticker}: index must be a DatetimeIndex.")
    if not df.index.is_monotonic_increasing:
        raise ValueError(f"{ticker}: index must be sorted ascending.")
    if df.index.has_duplicates:
        raise ValueError(f"{ticker}: index contains duplicate dates.")

    required = REQUIRED_YIELD_COLUMNS if is_yield else REQUIRED_ETF_COLUMNS
    missing_cols = [col for col in required if col not in df.columns]
    if missing_cols:
        raise ValueError(f"{ticker}: missing required columns {missing_cols}.")

    price_col = "close" if is_yield else "adj_close"
    series = df[price_col]
    if series.isna().any():
        nan_count = int(series.isna().sum())
        raise ValueError(f"{ticker}: found {nan_count} NaN values in '{price_col}'.")

    if is_yield:
        if (series < -5.0).any() or (series > 30.0).any():
            raise ValueError(
                f"{ticker}: yield values outside plausible range [-5%, 30%]: "
                f"[{float(series.min()):.4f}, {float(series.max()):.4f}]."
            )
    else:
        for col in ("open", "high", "low", "close", "adj_close"):
            if df[col].isna().any():
                raise ValueError(f"{ticker}: NaN values found in '{col}'.")
            if (df[col] <= 0).any():
                raise ValueError(f"{ticker}: non-positive values found in '{col}'.")

    years = complete_calendar_years(df)
    return {
        "ticker": ticker,
        "start_date": df.index.min().strftime("%Y-%m-%d"),
        "end_date": df.index.max().strftime("%Y-%m-%d"),
        "row_count": int(len(df)),
        "complete_years": years,
        "complete_year_count": len(years),
        "min_value": float(series.min()),
        "max_value": float(series.max()),
    }


def fetch_ticker_history(
    ticker: str,
    start: str = "1990-01-01",
    end: str | None = None,
) -> pd.DataFrame:
    """Download daily OHLCV + adjusted close + corporate actions from yfinance."""
    raw = yf.Ticker(ticker).history(
        start=start,
        end=end,
        interval="1d",
        auto_adjust=False,
        actions=True,
    )
    df = normalize_history_frame(raw)
    # Drop rows where core close/adj_close is missing (occasionally present on holiday stubs).
    price_col = "close" if ticker.startswith("^") else "adj_close"
    return df.dropna(subset=[price_col])


def load_manifest(manifest_path: Path = DEFAULT_MANIFEST_PATH) -> dict[str, Any]:
    """Load data/manifest.json, creating the base structure if missing."""
    if not manifest_path.exists():
        return {"schema_version": 1, "datasets": {}}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def save_raw_dataset(
    df: pd.DataFrame,
    ticker: str,
    *,
    raw_dir: Path = DEFAULT_RAW_DIR,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    source: str = "yfinance",
    is_yield: bool = False,
) -> dict[str, Any]:
    """Validate, persist raw CSV to data/raw/, and record SHA-256 provenance in manifest.json."""
    stats = validate_price_frame(df, ticker, is_yield=is_yield)

    raw_dir.mkdir(parents=True, exist_ok=True)
    filename = safe_ticker_filename(ticker)
    csv_path = raw_dir / filename
    df.to_csv(csv_path, date_format="%Y-%m-%d", lineterminator="\n")

    sha256 = compute_sha256(csv_path)
    fetched_at = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    years = stats["complete_years"]

    entry: dict[str, Any] = {
        "ticker": ticker,
        "file": f"data/raw/{filename}",
        "source": source,
        "is_yield": is_yield,
        "start_date": stats["start_date"],
        "end_date": stats["end_date"],
        "complete_years": {
            "first": years[0] if years else None,
            "last": years[-1] if years else None,
            "count": len(years),
        },
        "fetched_at": fetched_at,
        "row_count": stats["row_count"],
        "sha256": sha256,
    }

    manifest = load_manifest(manifest_path)
    manifest.setdefault("schema_version", 1)
    manifest.setdefault("datasets", {})
    manifest["datasets"][ticker] = entry
    manifest["datasets"] = dict(sorted(manifest["datasets"].items()))

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return entry


def load_raw_dataset(
    ticker: str,
    *,
    raw_dir: Path = DEFAULT_RAW_DIR,
    manifest_path: Path | None = DEFAULT_MANIFEST_PATH,
    verify_hash: bool = True,
) -> pd.DataFrame:
    """Load a cached raw CSV dataset and optionally verify its SHA-256 against manifest.json."""
    csv_path = raw_dir / safe_ticker_filename(ticker)
    if not csv_path.exists():
        raise FileNotFoundError(f"Raw dataset not found for {ticker}: {csv_path}")

    if verify_hash and manifest_path is not None:
        manifest = load_manifest(manifest_path)
        entry = manifest.get("datasets", {}).get(ticker)
        if entry is None:
            raise ValueError(f"{ticker} not found in manifest at {manifest_path}.")
        actual_hash = compute_sha256(csv_path)
        expected_hash = entry["sha256"]
        if actual_hash != expected_hash:
            raise ValueError(
                f"SHA-256 mismatch for {ticker}: expected {expected_hash}, got {actual_hash}."
            )

    df = pd.read_csv(csv_path, parse_dates=["date"], index_col="date")
    df.index = pd.DatetimeIndex(df.index).normalize()
    df.index.name = "date"
    return df.sort_index()


def fetch_and_store_all(
    tickers: tuple[str, ...] = DEFAULT_ASSUMPTIONS.tickers,
    risk_free_ticker: str = DEFAULT_ASSUMPTIONS.risk_free_ticker,
    raw_dir: Path = DEFAULT_RAW_DIR,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
) -> dict[str, dict[str, Any]]:
    """Fetch, validate, and store all configured ETF and risk-free series."""
    results: dict[str, dict[str, Any]] = {}
    for ticker in tickers:
        df = fetch_ticker_history(ticker)
        results[ticker] = save_raw_dataset(
            df,
            ticker,
            raw_dir=raw_dir,
            manifest_path=manifest_path,
            is_yield=False,
        )

    rf_df = fetch_ticker_history(risk_free_ticker)
    results[risk_free_ticker] = save_raw_dataset(
        rf_df,
        risk_free_ticker,
        raw_dir=raw_dir,
        manifest_path=manifest_path,
        is_yield=True,
    )
    return results


def main() -> None:
    """CLI entrypoint to download datasets and update data/manifest.json."""
    results = fetch_and_store_all()
    for ticker, entry in results.items():
        cy = entry["complete_years"]
        print(
            f"{ticker}: {entry['row_count']} rows "
            f"({entry['start_date']} to {entry['end_date']}), "
            f"complete years {cy['first']}-{cy['last']} ({cy['count']}), "
            f"sha256={entry['sha256'][:12]}..."
        )


if __name__ == "__main__":
    main()
