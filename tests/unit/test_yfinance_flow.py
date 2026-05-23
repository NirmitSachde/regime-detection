"""Batching + row normalization logic in the yfinance flow."""

from __future__ import annotations

from datetime import date

from regime.ingestion.yfinance_flow import _chunked, _norm_row


def test_chunked_even() -> None:
    out = _chunked(list(range(10)), 3)  # type: ignore[arg-type]
    assert [len(c) for c in out] == [3, 3, 3, 1]


def test_chunked_empty() -> None:
    assert _chunked([], 5) == []


def test_chunked_batch_size_50() -> None:
    out = _chunked(["T" + str(i) for i in range(120)], 50)
    assert [len(c) for c in out] == [50, 50, 20]


def test_norm_row_uses_adj_close_fallback() -> None:
    r = {
        "Date": date(2024, 1, 2),
        "Open": 100.0,
        "High": 101.0,
        "Low": 99.0,
        "Close": 100.5,
        "Volume": 1_000.0,
    }
    out = _norm_row("aapl", r)
    assert out["ticker"] == "AAPL"
    assert out["adj_close"] == 100.5


def test_norm_row_handles_zero_volume() -> None:
    r = {
        "Date": date(2024, 1, 2),
        "Open": 100.0,
        "High": 101.0,
        "Low": 99.0,
        "Close": 100.5,
        "Adj Close": 100.5,
        "Volume": 0.0,
    }
    out = _norm_row("AAPL", r)
    assert out["volume"] == 0.0
