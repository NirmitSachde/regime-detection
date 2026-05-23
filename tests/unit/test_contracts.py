"""Pydantic contract validation for raw rows."""

from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from regime.ingestion.contracts import IngestionRunStats, MacroSeriesPoint, OHLCVBar


def test_ohlcv_happy_path() -> None:
    bar = OHLCVBar(
        ticker="aapl",
        date=date(2024, 1, 2),
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.5,
        adj_close=100.5,
        volume=1_000_000,
    )
    assert bar.ticker == "AAPL"


def test_ohlcv_rejects_nonpositive_price() -> None:
    with pytest.raises(ValidationError):
        OHLCVBar(
            ticker="AAPL",
            date=date(2024, 1, 2),
            open=0.0,
            high=1.0,
            low=1.0,
            close=1.0,
            adj_close=1.0,
            volume=1.0,
        )


def test_ohlcv_rejects_negative_volume() -> None:
    with pytest.raises(ValidationError):
        OHLCVBar(
            ticker="AAPL",
            date=date(2024, 1, 2),
            open=1.0,
            high=1.0,
            low=1.0,
            close=1.0,
            adj_close=1.0,
            volume=-1.0,
        )


def test_macro_allows_missing_value() -> None:
    p = MacroSeriesPoint(series_id="DGS10", date=date(2024, 1, 2), value=None)
    assert p.value is None


def test_macro_uppercases_series_id() -> None:
    p = MacroSeriesPoint(series_id="dgs10", date=date(2024, 1, 2), value=4.2)
    assert p.series_id == "DGS10"


def test_ingestion_stats_serialization() -> None:
    s = IngestionRunStats(
        flow="ingest-yfinance",
        started_at="2024-01-01T00:00:00+00:00",
        finished_at="2024-01-01T00:01:00+00:00",
        rows_in=100,
        rows_written=100,
        files_written=4,
        bytes_written=1024,
        new_or_changed=True,
    )
    d = s.model_dump()
    assert d["rows_written"] == 100
