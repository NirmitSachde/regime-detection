"""End-to-end: ingest → bootstrap warehouse → dbt build → train → backtest.

Usage:
    uv run python scripts/run_pipeline.py [--skip-ingest] [--skip-train] [--skip-backtest]
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import typer

from regime.ingestion.fred_flow import ingest_macro_flow
from regime.ingestion.yfinance_flow import ingest_prices_flow
from regime.logging import get_logger
from regime.warehouse import bootstrap

log = get_logger(__name__)
app = typer.Typer(no_args_is_help=False)

REPO_ROOT = Path(__file__).resolve().parent.parent


@app.command()
def main(
    skip_ingest: bool = False,
    skip_dbt: bool = False,
    skip_train: bool = False,
    skip_backtest: bool = False,
) -> None:
    if not skip_ingest:
        log.info("pipeline.ingest")
        ingest_prices_flow()
        ingest_macro_flow()

    log.info("pipeline.warehouse_bootstrap")
    bootstrap()

    if not skip_dbt:
        log.info("pipeline.dbt_build")
        rc = subprocess.run(
            ["dbt", "build", "--profiles-dir", "."],
            cwd=REPO_ROOT / "dbt",
            check=False,
        ).returncode
        if rc != 0:
            log.error("pipeline.dbt_build.failed", returncode=rc)
            sys.exit(rc)

    if not skip_train:
        log.info("pipeline.train")
        from regime.models.train import train_hmm, train_lgbm

        train_hmm()
        train_lgbm()

    if not skip_backtest:
        log.info("pipeline.backtest")
        from regime.backtest.engine import run_all_backtests

        run_all_backtests()

    log.info("pipeline.done")


if __name__ == "__main__":
    app()
