"""Prefect 3 flow: pull macro series from FRED, write partitioned Parquet."""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import cast

import polars as pl
from fredapi import Fred
from prefect import flow, task
from prefect.tasks import exponential_backoff

from regime.config import get_settings
from regime.ingestion.contracts import IngestionRunStats
from regime.ingestion.storage import macro_partition_path, write_partition_idempotent
from regime.logging import get_logger

log = get_logger(__name__)

# Default macro panel — extend in handoff Phase 5 as research dictates.
DEFAULT_SERIES: tuple[str, ...] = (
    "DGS10",        # 10Y Treasury
    "DGS2",         # 2Y Treasury
    "T10Y2Y",       # 10Y - 2Y spread
    "VIXCLS",       # VIX close
    "DTWEXBGS",     # Trade-weighted dollar
    "CPIAUCSL",     # CPI
    "UNRATE",       # Unemployment
    "FEDFUNDS",     # Fed funds rate
    "BAMLH0A0HYM2", # HY OAS
    "DCOILWTICO",   # WTI crude
)


@task(
    retries=3,
    retry_delay_seconds=exponential_backoff(backoff_factor=4),
    retry_jitter_factor=0.2,
)
def fetch_series(series_id: str, start: str) -> pl.DataFrame:
    settings = get_settings()
    if not settings.fred_api_key:
        raise RuntimeError(
            "FRED_API_KEY is not set. Get a free key at "
            "https://fredaccount.stlouisfed.org/apikeys and put it in .env"
        )
    fred = Fred(api_key=settings.fred_api_key)
    s = fred.get_series(series_id, observation_start=start)
    if s is None or s.empty:
        return pl.DataFrame()

    df = pl.DataFrame(
        {
            "series_id": [series_id.upper()] * len(s),
            "date": [d.date() if hasattr(d, "date") else d for d in s.index],
            "value": [None if (v != v) else float(v) for v in s.values],  # noqa: PLR0124
        }
    )
    log.info("fred.fetched", series=series_id, rows=df.height)
    return df


@task
def write_macro(df: pl.DataFrame) -> tuple[int, int]:
    if df.height == 0:
        return 0, 0
    settings = get_settings()
    df = df.with_columns(pl.col("date").dt.year().alias("_year"))
    files = 0
    total_bytes = 0
    for (series_id, year), group in df.group_by(["series_id", "_year"]):
        out = macro_partition_path(settings.macro_dir, cast(str, series_id), cast(int, year))
        part = group.drop(["_year"])
        wrote, nbytes = write_partition_idempotent(part, out, sort_by=["date"])
        if wrote:
            files += 1
            total_bytes += nbytes
    return files, total_bytes


@flow(name="ingest-fred")
def ingest_macro_flow(
    series: list[str] | None = None,
    start: str | None = None,
) -> IngestionRunStats:
    settings = get_settings()
    series = series or list(DEFAULT_SERIES)
    start = start or settings.history_start

    started = datetime.now(UTC).isoformat()
    log.info("ingest.macro.start", n_series=len(series), start=start)

    futures = [fetch_series.submit(s, start) for s in series]
    parts = [f.result() for f in futures]
    parts = [p for p in parts if p.height > 0]

    if not parts:
        log.warning("ingest.macro.empty")
        return IngestionRunStats(
            flow="ingest-fred",
            started_at=started,
            finished_at=datetime.now(UTC).isoformat(),
            rows_in=0,
            rows_written=0,
            files_written=0,
            bytes_written=0,
            new_or_changed=False,
        )

    combined = pl.concat(parts, how="vertical_relaxed")
    files, nbytes = write_macro(combined)

    stats = IngestionRunStats(
        flow="ingest-fred",
        started_at=started,
        finished_at=datetime.now(UTC).isoformat(),
        rows_in=combined.height,
        rows_written=combined.height if files > 0 else 0,
        files_written=files,
        bytes_written=nbytes,
        new_or_changed=files > 0,
    )
    log.info("ingest.macro.done", **stats.model_dump())
    return stats


# Convenience for nightly-runner callers
def schedule_nightly() -> None:  # pragma: no cover - infra glue
    """Register both flows on a 22:00 UTC cron via Prefect deployments.

    Apply with: `uv run python -m regime.ingestion.fred_flow --schedule`
    Or use `prefect deploy` with `prefect.yaml` (see scripts/deploy_flows.py).
    """
    from prefect.schedules import Cron  # local import to keep top-level light

    from regime.ingestion.yfinance_flow import ingest_prices_flow

    ingest_prices_flow.deploy(
        name="nightly-prices",
        work_pool_name="default",
        schedules=[Cron("0 22 * * *", timezone="UTC")],
    )
    ingest_macro_flow.deploy(
        name="nightly-macro",
        work_pool_name="default",
        schedules=[Cron("15 22 * * *", timezone="UTC")],
    )


_ = date  # silence unused-import for stub
