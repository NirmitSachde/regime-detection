"""Strip path-dependent VIEWs from data/warehouse.duckdb before shipping.

dbt creates `raw.prices`, `raw.macro`, and `main_staging.*` as VIEWs that
hard-code the absolute filesystem path to the local `data/raw/` parquet
partitions (set when the warehouse was bootstrapped on the dev's machine).
On Render (or anywhere else without that path), opening the warehouse is
fine but any code that touches those views fails.

The materialised intermediate + marts tables — which the API actually
reads — are independent of the raw partitions, so dropping the views is
safe. Run this after `make real-data` and before `make api-image`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb

WAREHOUSE = Path("data/warehouse.duckdb")

# Schemas / objects that depend on absolute paths
VIEWS_TO_DROP = [
    ("raw", "prices"),
    ("raw", "macro"),
    ("main_staging", "stg_prices"),
    ("main_staging", "stg_macro"),
]
SCHEMAS_TO_DROP = ["raw", "main_staging"]


def main() -> None:
    if not WAREHOUSE.exists():
        print(f"warehouse not found at {WAREHOUSE} — nothing to clean", file=sys.stderr)
        sys.exit(0)

    con = duckdb.connect(str(WAREHOUSE))
    try:
        for schema, name in VIEWS_TO_DROP:
            con.execute(f'DROP VIEW IF EXISTS {schema}."{name}"')
            print(f"  dropped view {schema}.{name}")
        for schema in SCHEMAS_TO_DROP:
            con.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
            print(f"  dropped schema {schema}")
        remaining = con.execute(
            """
            select table_schema, table_name, table_type
            from information_schema.tables
            where table_schema not in ('information_schema', 'pg_catalog')
            order by table_schema, table_name
            """
        ).fetchall()
        print("\nRemaining objects:")
        for r in remaining:
            print(f"  {r[2]:12s} {r[0]}.{r[1]}")
    finally:
        con.close()

    size_mb = WAREHOUSE.stat().st_size / (1024 * 1024)
    print(f"\nWarehouse size: {size_mb:.1f}MB")


if __name__ == "__main__":
    main()
