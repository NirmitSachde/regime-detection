"""DuckDB warehouse bootstrap.

DuckDB reads Parquet partitions natively via `read_parquet(..., hive_partitioning=true)`.
We create *views* over the raw zone rather than copying — zero-copy analytics.
"""

from __future__ import annotations

import duckdb

from regime.config import get_settings
from regime.logging import get_logger

log = get_logger(__name__)


def connect() -> duckdb.DuckDBPyConnection:
    settings = get_settings()
    settings.duckdb_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(settings.duckdb_path))
    return con


def bootstrap() -> None:
    """Create raw-zone schemas and views. Idempotent.

    Uses absolute glob paths so that dbt (which runs from the dbt/ subdir)
    can still resolve the parquet partitions when it opens the same
    warehouse file.
    """
    settings = get_settings()
    con = connect()
    try:
        con.execute("CREATE SCHEMA IF NOT EXISTS raw")

        # Resolve the base dirs first (Path.resolve doesn't handle glob chars);
        # then append the glob suffix as a string so DuckDB sees an absolute pattern.
        prices_glob = f"{settings.prices_dir.resolve()}/**/*.parquet"
        macro_glob = f"{settings.macro_dir.resolve()}/**/*.parquet"

        con.execute(
            f"""
            CREATE OR REPLACE VIEW raw.prices AS
            SELECT * FROM read_parquet(
                '{prices_glob}',
                hive_partitioning = true,
                union_by_name = true
            )
            """
        )
        con.execute(
            f"""
            CREATE OR REPLACE VIEW raw.macro AS
            SELECT * FROM read_parquet(
                '{macro_glob}',
                hive_partitioning = true,
                union_by_name = true
            )
            """
        )
        log.info("warehouse.bootstrap.done", duckdb=str(settings.duckdb_path))
    finally:
        con.close()


if __name__ == "__main__":
    bootstrap()
