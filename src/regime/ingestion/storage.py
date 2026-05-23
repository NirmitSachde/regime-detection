"""Parquet partitioned writer with content-hashed idempotency.

Layout:
    data/raw/prices/ticker=AAPL/year=2024/month=01/part.parquet
    data/raw/macro/series_id=DGS10/year=2024/part.parquet

A re-run that produces the same content writes no bytes — the file is only
replaced if the sha256 of the serialized table changes.
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

if TYPE_CHECKING:
    from collections.abc import Iterable


def _hash_bytes(buf: bytes) -> str:
    return hashlib.sha256(buf).hexdigest()


def _table_to_bytes(table: pa.Table) -> bytes:
    sink = io.BytesIO()
    pq.write_table(table, sink, compression="zstd", compression_level=3)
    return sink.getvalue()


def _read_hash(path: Path) -> str | None:
    if not path.exists():
        return None
    return _hash_bytes(path.read_bytes())


def write_partition_idempotent(
    df: pl.DataFrame,
    out_path: Path,
    sort_by: Iterable[str],
) -> tuple[bool, int]:
    """Write `df` to `out_path` only if content differs from existing file.

    Returns:
        (was_written, bytes_written). bytes_written is 0 on no-op.
    """
    if df.height == 0:
        return False, 0

    sorted_df = df.sort(list(sort_by))
    table = sorted_df.to_arrow()
    new_bytes = _table_to_bytes(table)
    new_hash = _hash_bytes(new_bytes)

    if _read_hash(out_path) == new_hash:
        return False, 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(new_bytes)
    return True, len(new_bytes)


def prices_partition_path(root: Path, ticker: str, year: int, month: int) -> Path:
    return (
        root
        / f"ticker={ticker.upper()}"
        / f"year={year}"
        / f"month={month:02d}"
        / "part.parquet"
    )


def macro_partition_path(root: Path, series_id: str, year: int) -> Path:
    return root / f"series_id={series_id.upper()}" / f"year={year}" / "part.parquet"
