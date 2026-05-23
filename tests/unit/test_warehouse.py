"""Warehouse bootstrap smoke test against synthetic raw zone."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import duckdb
import polars as pl
import pytest

from regime.ingestion.storage import (
    macro_partition_path,
    prices_partition_path,
    write_partition_idempotent,
)


@pytest.fixture
def synth_raw(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Build a tiny raw zone and point Settings at it."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("RAW_DIR", str(tmp_path / "raw"))
    monkeypatch.setenv("DUCKDB_PATH", str(tmp_path / "warehouse.duckdb"))

    # Clear settings cache so env var overrides take effect
    from regime.config import get_settings

    get_settings.cache_clear()

    prices = pl.DataFrame(
        {
            "ticker": ["SPY"] * 3,
            "date": [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)],
            "open": [470.0, 471.0, 472.0],
            "high": [472.0, 473.0, 474.0],
            "low": [469.0, 470.0, 471.0],
            "close": [471.0, 472.0, 473.0],
            "adj_close": [471.0, 472.0, 473.0],
            "volume": [1e8, 1.1e8, 1.2e8],
        }
    )
    macro = pl.DataFrame(
        {
            "series_id": ["VIXCLS"] * 3,
            "date": [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)],
            "value": [14.0, 14.5, 15.0],
        }
    )

    settings = get_settings()
    write_partition_idempotent(
        prices,
        prices_partition_path(settings.prices_dir, "SPY", 2024, 1),
        sort_by=["date"],
    )
    write_partition_idempotent(
        macro,
        macro_partition_path(settings.macro_dir, "VIXCLS", 2024),
        sort_by=["date"],
    )
    return tmp_path


def test_bootstrap_creates_raw_views(synth_raw: Path) -> None:
    from regime.warehouse import bootstrap, connect

    bootstrap()
    con = connect()
    try:
        n_prices = con.execute("select count(*) from raw.prices").fetchone()
        n_macro = con.execute("select count(*) from raw.macro").fetchone()
    finally:
        con.close()
    assert n_prices is not None and n_prices[0] == 3
    assert n_macro is not None and n_macro[0] == 3


def test_bootstrap_is_idempotent(synth_raw: Path) -> None:
    from regime.warehouse import bootstrap, connect

    bootstrap()
    bootstrap()  # second call must not raise
    con = connect()
    try:
        res = con.execute(
            "select table_name from information_schema.tables where table_schema='raw'"
        ).fetchall()
    finally:
        con.close()
    names = sorted(r[0] for r in res)
    assert names == ["macro", "prices"]


def test_duckdb_can_read_hive_partitions(synth_raw: Path) -> None:
    from regime.config import get_settings

    settings = get_settings()
    glob = str(settings.prices_dir / "**" / "*.parquet")
    con = duckdb.connect(":memory:")
    n = con.execute(
        f"select count(*) from read_parquet('{glob}', hive_partitioning=true)"
    ).fetchone()
    assert n is not None and n[0] == 3
