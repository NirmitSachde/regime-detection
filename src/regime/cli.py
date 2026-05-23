"""Top-level CLI entry point. Sub-CLIs live in their respective subpackages."""

from __future__ import annotations

import typer

from regime import __version__
from regime.backtest.cli import app as backtest_app
from regime.ingestion.cli import app as ingest_app
from regime.models.train import app as train_app

app = typer.Typer(
    name="regime",
    help="Adaptive Market Regime Detection & Backtesting Platform.",
    no_args_is_help=True,
    add_completion=False,
)

app.add_typer(ingest_app, name="ingest", help="Ingestion flows (yfinance, FRED).")
app.add_typer(train_app, name="train", help="Train regime models (HMM, LightGBM).")
app.add_typer(backtest_app, name="backtest", help="Run vectorbt backtests.")


@app.command()
def version() -> None:
    """Print package version."""
    typer.echo(__version__)


if __name__ == "__main__":
    app()
