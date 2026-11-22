"""Offline unit tests for modeling assumptions, data validation, and manifest provenance."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from etf_strategy.config import DEFAULT_ASSUMPTIONS
from etf_strategy.data import (
    complete_calendar_years,
    filter_complete_years,
    load_manifest,
    load_raw_dataset,
    normalize_history_frame,
    safe_ticker_filename,
    save_raw_dataset,
    validate_price_frame,
)


def _make_etf_frame(start: str, end: str, base_price: float = 100.0) -> pd.DataFrame:
    dates = pd.bdate_range(start=start, end=end)
    prices = base_price + np.arange(len(dates), dtype=float) * 0.1
    return pd.DataFrame(
        {
            "open": prices,
            "high": prices * 1.01,
            "low": prices * 0.99,
            "close": prices,
            "adj_close": prices,
            "volume": 1_000_000.0,
            "dividends": 0.0,
            "stock_splits": 0.0,
        },
        index=pd.DatetimeIndex(dates, name="date"),
    )


def test_default_assumptions() -> None:
    assert DEFAULT_ASSUMPTIONS.tickers == ("SPY", "QQQ")
    assert DEFAULT_ASSUMPTIONS.risk_free_ticker == "^IRX"
    assert DEFAULT_ASSUMPTIONS.annual_budget == 12_000.0
    assert DEFAULT_ASSUMPTIONS.fractional_shares is True
    assert DEFAULT_ASSUMPTIONS.commission_per_trade == 0.0
    assert DEFAULT_ASSUMPTIONS.execution_price_col == "adj_close"


def test_safe_ticker_filename() -> None:
    assert safe_ticker_filename("SPY") == "SPY.csv"
    assert safe_ticker_filename("^IRX") == "IRX.csv"
    assert safe_ticker_filename("BRK.B") == "BRK_B.csv"


def test_normalize_history_frame_handles_tz_and_columns() -> None:
    dates = pd.bdate_range("2022-01-03", periods=5, tz="America/New_York")
    raw = pd.DataFrame(
        {
            "Open": [100.0] * 5,
            "High": [101.0] * 5,
            "Low": [99.0] * 5,
            "Close": [100.5] * 5,
            "Adj Close": [98.0] * 5,
            "Volume": [1000] * 5,
            "Dividends": [0.0] * 5,
            "Stock Splits": [0.0] * 5,
        },
        index=dates,
    )
    norm = normalize_history_frame(raw)
    assert norm.index.name == "date"
    assert norm.index.tz is None
    assert list(norm.columns) == [
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
        "dividends",
        "stock_splits",
    ]


def test_filter_complete_years_drops_partial_boundary_years() -> None:
    df = _make_etf_frame("2019-06-01", "2022-05-15")
    assert complete_calendar_years(df) == [2020, 2021]

    filtered = filter_complete_years(df)
    assert sorted(filtered.index.year.unique().tolist()) == [2020, 2021]


def test_validate_price_frame_rejects_bad_data() -> None:
    valid = _make_etf_frame("2021-01-04", "2021-12-31")
    stats = validate_price_frame(valid, "SPY")
    assert stats["complete_years"] == [2021]

    with_nan = valid.copy()
    with_nan.iloc[5, with_nan.columns.get_loc("adj_close")] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        validate_price_frame(with_nan, "SPY")

    non_positive = valid.copy()
    non_positive.iloc[0, non_positive.columns.get_loc("adj_close")] = 0.0
    with pytest.raises(ValueError, match="non-positive"):
        validate_price_frame(non_positive, "SPY")

    unsorted = valid.iloc[::-1]
    with pytest.raises(ValueError, match="sorted ascending"):
        validate_price_frame(unsorted, "SPY")


def test_validate_yield_frame() -> None:
    dates = pd.bdate_range("2021-01-04", "2021-12-31")
    rf = pd.DataFrame(
        {"close": 4.25, "adj_close": 4.25},
        index=pd.DatetimeIndex(dates, name="date"),
    )
    stats = validate_price_frame(rf, "^IRX", is_yield=True)
    assert stats["row_count"] == len(dates)

    bad_rf = rf.copy()
    bad_rf.iloc[0, bad_rf.columns.get_loc("close")] = 45.0
    with pytest.raises(ValueError, match="plausible range"):
        validate_price_frame(bad_rf, "^IRX", is_yield=True)


def test_save_and_load_raw_dataset_with_sha256_verification(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    manifest_path = tmp_path / "manifest.json"
    df = _make_etf_frame("2020-01-02", "2021-12-31")

    entry = save_raw_dataset(df, "SPY", raw_dir=raw_dir, manifest_path=manifest_path)
    assert entry["ticker"] == "SPY"
    assert entry["row_count"] == len(df)
    assert entry["complete_years"] == {"first": 2020, "last": 2021, "count": 2}
    assert len(entry["sha256"]) == 64

    manifest = load_manifest(manifest_path)
    assert manifest["datasets"]["SPY"]["sha256"] == entry["sha256"]

    loaded = load_raw_dataset("SPY", raw_dir=raw_dir, manifest_path=manifest_path, verify_hash=True)
    pd.testing.assert_frame_equal(loaded, df, check_freq=False)

    # Tampering with raw CSV must fail hash verification.
    csv_path = raw_dir / "SPY.csv"
    csv_path.write_text(csv_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_raw_dataset("SPY", raw_dir=raw_dir, manifest_path=manifest_path, verify_hash=True)
