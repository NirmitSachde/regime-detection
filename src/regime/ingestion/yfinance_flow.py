"""Prefect 3 flow: pull daily OHLCV from yfinance, write partitioned Parquet."""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any, cast

import polars as pl
import yfinance as yf
from prefect import flow, task
from prefect.tasks import exponential_backoff
from tenacity import retry, stop_after_attempt, wait_exponential

from regime.config import get_settings
from regime.ingestion.contracts import IngestionRunStats
from regime.ingestion.storage import prices_partition_path, write_partition_idempotent
from regime.logging import get_logger

log = get_logger(__name__)

BATCH_SIZE = 50


def _chunked(seq: list[str], n: int) -> list[list[str]]:
    return [seq[i : i + n] for i in range(0, len(seq), n)]


@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    reraise=True,
)
def _download_batch(tickers: list[str], start: str, end: str) -> pl.DataFrame:
    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        threads=True,
        group_by="ticker",
    )
    if raw is None or raw.empty:
        return pl.DataFrame()

    # yfinance returns a multi-index column frame for multi-ticker requests
    rows: list[dict[str, Any]] = []
    if len(tickers) == 1:
        single = tickers[0]
        df = raw.reset_index()
        for r in df.to_dict(orient="records"):
            rows.append(_norm_row(single, r))
    else:
        for tkr in tickers:
            if tkr not in raw.columns.get_level_values(0):
                continue
            sub = raw[tkr].reset_index().dropna(how="all")
            for r in sub.to_dict(orient="records"):
                rows.append(_norm_row(tkr, r))

    if not rows:
        return pl.DataFrame()
    return pl.from_dicts(rows)


def _norm_row(ticker: str, r: dict[str, Any]) -> dict[str, Any]:
    d = r.get("Date") or r.get("Datetime") or r.get("index")
    if d is not None and hasattr(d, "date"):
        d = d.date()
    return {
        "ticker": ticker.upper(),
        "date": d,
        "open": float(r.get("Open", 0.0) or 0.0),
        "high": float(r.get("High", 0.0) or 0.0),
        "low": float(r.get("Low", 0.0) or 0.0),
        "close": float(r.get("Close", 0.0) or 0.0),
        "adj_close": float(r.get("Adj Close", r.get("Close", 0.0)) or 0.0),
        "volume": float(r.get("Volume", 0.0) or 0.0),
    }


@task(
    retries=3,
    retry_delay_seconds=exponential_backoff(backoff_factor=4),
    retry_jitter_factor=0.2,
    log_prints=False,
)
def fetch_batch(tickers: list[str], start: str, end: str) -> pl.DataFrame:
    log.info("yf.fetch_batch", n=len(tickers), start=start, end=end)
    df = _download_batch(tickers, start, end)
    log.info("yf.fetch_batch.done", n=len(tickers), rows=df.height)
    return df


@task
def write_prices(df: pl.DataFrame) -> tuple[int, int]:
    """Write per (ticker, year, month) partitions. Returns (files_written, bytes)."""
    if df.height == 0:
        return 0, 0
    settings = get_settings()

    # Drop bad rows that would fail downstream contracts
    df = df.filter(
        (pl.col("open") > 0)
        & (pl.col("high") > 0)
        & (pl.col("low") > 0)
        & (pl.col("close") > 0)
        & (pl.col("adj_close") > 0)
        & (pl.col("volume") >= 0)
    )

    df = df.with_columns(
        pl.col("date").dt.year().alias("_year"),
        pl.col("date").dt.month().alias("_month"),
    )

    files = 0
    total_bytes = 0
    for (ticker, year, month), group in df.group_by(["ticker", "_year", "_month"]):
        out = prices_partition_path(
            settings.prices_dir,
            cast("str", ticker),
            cast("int", year),
            cast("int", month),
        )
        part = group.drop(["_year", "_month"])
        wrote, nbytes = write_partition_idempotent(part, out, sort_by=["date"])
        if wrote:
            files += 1
            total_bytes += nbytes
    return files, total_bytes


@flow(name="ingest-yfinance")
def ingest_prices_flow(
    tickers: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
) -> IngestionRunStats:
    settings = get_settings()
    tickers = tickers or settings.universe_list
    start = start or settings.history_start
    end = end or date.today().isoformat()

    started = datetime.now(UTC).isoformat()
    log.info("ingest.prices.start", n_tickers=len(tickers), start=start, end=end)

    futures = [fetch_batch.submit(b, start, end) for b in _chunked(tickers, BATCH_SIZE)]
    parts = [f.result() for f in futures]
    parts = [p for p in parts if p.height > 0]

    if not parts:
        log.warning("ingest.prices.empty")
        return IngestionRunStats(
            flow="ingest-yfinance",
            started_at=started,
            finished_at=datetime.now(UTC).isoformat(),
            rows_in=0,
            rows_written=0,
            files_written=0,
            bytes_written=0,
            new_or_changed=False,
        )

    combined = pl.concat(parts, how="vertical_relaxed")
    files, nbytes = write_prices(combined)

    stats = IngestionRunStats(
        flow="ingest-yfinance",
        started_at=started,
        finished_at=datetime.now(UTC).isoformat(),
        rows_in=combined.height,
        rows_written=combined.height if files > 0 else 0,
        files_written=files,
        bytes_written=nbytes,
        new_or_changed=files > 0,
    )
    log.info("ingest.prices.done", **stats.model_dump())
    return stats
