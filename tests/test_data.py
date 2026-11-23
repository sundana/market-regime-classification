from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from market_regime_classification.data import (
    _extract_close_prices,
    _validate_price_frame,
    load_csv,
)


def _write(tmp_path: Path, frame: pd.DataFrame) -> Path:
    path = tmp_path / "prices.csv"
    frame.to_csv(path, index=False)
    return path


def _frame(**columns: object) -> pd.DataFrame:
    data = {"date": pd.bdate_range("2020-01-01", periods=len(next(iter(columns.values()))))}
    data.update(columns)
    return pd.DataFrame(data)


def test_load_csv_sorts_and_returns_positive_prices(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        _frame(
            date=["2020-01-03", "2020-01-01", "2020-01-02"],
            A=[3.0, 1.0, 2.0],
            B=[30.0, 10.0, 20.0],
        ),
    )
    prices = load_csv(path)

    assert list(prices.columns) == ["A", "B"]
    assert prices.index.is_monotonic_increasing
    assert (prices > 0).all().all()
    assert prices.index.name == "date"


def test_load_csv_rejects_duplicate_dates(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        _frame(date=["2020-01-01", "2020-01-01", "2020-01-02"], A=[1.0, 2.0, 3.0]),
    )
    with pytest.raises(ValueError, match="duplicate dates"):
        load_csv(path)


def test_load_csv_rejects_missing_date_column(tmp_path: Path) -> None:
    path = _write(tmp_path, _frame(date=["2020-01-01"], A=[1.0]))
    with pytest.raises(KeyError, match="date column"):
        load_csv(path, date_col="when")


def test_load_csv_rejects_no_numeric_columns(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        pd.DataFrame({"date": ["2020-01-01", "2020-01-02"], "name": ["a", "b"]}),
    )
    with pytest.raises(ValueError, match="no numeric price columns"):
        load_csv(path)


def test_load_csv_rejects_non_positive_prices(tmp_path: Path) -> None:
    path = _write(tmp_path, _frame(date=["2020-01-01", "2020-01-02"], A=[1.0, 0.0]))
    with pytest.raises(ValueError, match="non-positive price"):
        load_csv(path)


def test_load_csv_rejects_infinite_prices(tmp_path: Path) -> None:
    path = _write(tmp_path, _frame(date=["2020-01-01", "2020-01-02"], A=[1.0, np.inf]))
    with pytest.raises(ValueError, match="infinite"):
        load_csv(path)


def test_load_csv_fills_short_gaps(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        _frame(
            date=pd.bdate_range("2020-01-01", periods=5),
            A=[1.0, np.nan, np.nan, 4.0, 5.0],
            B=[10.0, 11.0, 12.0, 13.0, 14.0],
        ),
    )
    prices = load_csv(path, max_stale_days=2)
    assert prices["A"].tolist() == [1.0, 1.0, 1.0, 4.0, 5.0]


def test_load_csv_rejects_long_gaps(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        _frame(
            date=pd.bdate_range("2020-01-01", periods=5),
            A=[1.0, np.nan, np.nan, np.nan, 5.0],
            B=[10.0, 11.0, 12.0, 13.0, 14.0],
        ),
    )
    with pytest.raises(ValueError, match="gaps longer than max_stale_days=2"):
        load_csv(path, max_stale_days=2)


def test_load_csv_zero_stale_days_disables_filling(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        _frame(
            date=pd.bdate_range("2020-01-01", periods=3),
            A=[1.0, np.nan, 3.0],
            B=[10.0, 11.0, 12.0],
        ),
    )
    with pytest.raises(ValueError, match="gaps longer than max_stale_days=0"):
        load_csv(path, max_stale_days=0)

    complete_dir = tmp_path / "complete"
    complete_dir.mkdir()
    complete = _write(
        complete_dir,
        _frame(
            date=pd.bdate_range("2020-01-01", periods=3),
            A=[1.0, 2.0, 3.0],
            B=[10.0, 11.0, 12.0],
        ),
    )
    assert len(load_csv(complete, max_stale_days=0)) == 3


def test_load_csv_trims_history_before_all_assets_exist(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        _frame(
            date=pd.bdate_range("2020-01-01", periods=4),
            A=[np.nan, np.nan, 3.0, 4.0],
            B=[10.0, 11.0, 12.0, 13.0],
        ),
    )
    prices = load_csv(path)
    assert len(prices) == 2
    assert prices.index[0] == pd.Timestamp("2020-01-03")


def test_load_csv_rejects_asset_with_no_prices(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        pd.DataFrame(
            {
                "date": pd.bdate_range("2020-01-01", periods=3),
                "A": [1.0, 2.0, 3.0],
                "B": [np.nan, np.nan, np.nan],
            }
        ),
    )
    with pytest.raises(ValueError, match="no usable prices"):
        load_csv(path)


def test_load_csv_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_csv(tmp_path / "missing.csv")


def test_validate_requires_datetime_index() -> None:
    with pytest.raises(TypeError, match="DatetimeIndex"):
        _validate_price_frame(pd.DataFrame({"A": [1.0, 2.0]}), source="test")


def test_validate_rejects_duplicate_columns() -> None:
    frame = pd.DataFrame(
        np.ones((2, 2)),
        columns=["A", "A"],
        index=pd.bdate_range("2020-01-01", periods=2),
    )
    with pytest.raises(ValueError, match="duplicate price columns"):
        _validate_price_frame(frame, source="test")


def test_validate_rejects_negative_stale_limit() -> None:
    frame = pd.DataFrame(
        {"A": [1.0, 2.0]}, index=pd.bdate_range("2020-01-01", periods=2)
    )
    with pytest.raises(ValueError, match="max_stale_days must be >= 0"):
        _validate_price_frame(frame, source="test", max_stale_days=-1)


def test_extract_close_prices_from_multiindex() -> None:
    columns = pd.MultiIndex.from_product([["Close", "Open"], ["SPY", "TLT"]])
    raw = pd.DataFrame(
        [[100.0, 200.0, 99.0, 199.0]],
        columns=columns,
        index=pd.DatetimeIndex(["2020-01-02"]),
    )
    prices = _extract_close_prices(raw, ["SPY", "TLT"])
    assert list(prices.columns) == ["SPY", "TLT"]
    assert prices.iloc[0].tolist() == [100.0, 200.0]


def test_extract_close_prices_from_flat_columns() -> None:
    raw = pd.DataFrame(
        [[99.0, 100.0]],
        columns=["Open", "Close"],
        index=pd.DatetimeIndex(["2020-01-02"]),
    )
    prices = _extract_close_prices(raw, ["SPY"])
    assert list(prices.columns) == ["SPY"]
    assert prices.iloc[0, 0] == 100.0


def test_extract_close_prices_reports_missing_ticker() -> None:
    columns = pd.MultiIndex.from_product([["Close"], ["SPY"]])
    raw = pd.DataFrame(
        [[100.0]], columns=columns, index=pd.DatetimeIndex(["2020-01-02"])
    )
    with pytest.raises(RuntimeError, match="did not return prices"):
        _extract_close_prices(raw, ["SPY", "TLT"])
