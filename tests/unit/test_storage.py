"""Idempotency + partitioning behavior for the raw-zone writer."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest

from regime.ingestion.storage import (
    macro_partition_path,
    prices_partition_path,
    write_partition_idempotent,
)


@pytest.fixture
def sample_prices() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ticker": ["AAPL", "AAPL", "AAPL"],
            "date": [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)],
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.5, 101.5, 102.5],
            "adj_close": [100.5, 101.5, 102.5],
            "volume": [1_000_000.0, 1_100_000.0, 1_200_000.0],
        }
    )


def test_partition_path_prices(tmp_path: Path) -> None:
    p = prices_partition_path(tmp_path, "aapl", 2024, 3)
    assert p == tmp_path / "ticker=AAPL" / "year=2024" / "month=03" / "part.parquet"


def test_partition_path_macro(tmp_path: Path) -> None:
    p = macro_partition_path(tmp_path, "dgs10", 2024)
    assert p == tmp_path / "series_id=DGS10" / "year=2024" / "part.parquet"


def test_write_creates_file(tmp_path: Path, sample_prices: pl.DataFrame) -> None:
    out = tmp_path / "part.parquet"
    wrote, nbytes = write_partition_idempotent(sample_prices, out, sort_by=["date"])
    assert wrote is True
    assert nbytes > 0
    assert out.exists()


def test_write_is_idempotent(tmp_path: Path, sample_prices: pl.DataFrame) -> None:
    """Re-running on the same data must produce zero new bytes."""
    out = tmp_path / "part.parquet"
    write_partition_idempotent(sample_prices, out, sort_by=["date"])
    wrote2, nbytes2 = write_partition_idempotent(sample_prices, out, sort_by=["date"])
    assert wrote2 is False
    assert nbytes2 == 0


def test_write_detects_content_change(tmp_path: Path, sample_prices: pl.DataFrame) -> None:
    out = tmp_path / "part.parquet"
    write_partition_idempotent(sample_prices, out, sort_by=["date"])
    bumped = sample_prices.with_columns(pl.col("close") + 0.01)
    wrote2, nbytes2 = write_partition_idempotent(bumped, out, sort_by=["date"])
    assert wrote2 is True
    assert nbytes2 > 0


def test_empty_input_is_noop(tmp_path: Path) -> None:
    out = tmp_path / "part.parquet"
    empty = pl.DataFrame(
        schema={
            "ticker": pl.String,
            "date": pl.Date,
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "adj_close": pl.Float64,
            "volume": pl.Float64,
        }
    )
    wrote, nbytes = write_partition_idempotent(empty, out, sort_by=["date"])
    assert wrote is False
    assert nbytes == 0
    assert not out.exists()


def test_row_order_does_not_affect_hash(tmp_path: Path, sample_prices: pl.DataFrame) -> None:
    """Sorted-on-write means input row order is irrelevant for idempotency."""
    out = tmp_path / "part.parquet"
    write_partition_idempotent(sample_prices, out, sort_by=["date"])
    shuffled = sample_prices.sample(fraction=1.0, shuffle=True, seed=7)
    wrote2, nbytes2 = write_partition_idempotent(shuffled, out, sort_by=["date"])
    assert wrote2 is False
    assert nbytes2 == 0
