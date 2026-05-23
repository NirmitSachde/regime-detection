"""CLI for ingestion flows. `uv run regime-ingest --help`."""

from __future__ import annotations

import typer

from regime.ingestion.fred_flow import ingest_macro_flow
from regime.ingestion.yfinance_flow import ingest_prices_flow

app = typer.Typer(name="ingest", help="Run ingestion flows.", no_args_is_help=True)


@app.command()
def prices(
    start: str | None = typer.Option(None, help="YYYY-MM-DD"),
    end: str | None = typer.Option(None, help="YYYY-MM-DD"),
    tickers: str | None = typer.Option(None, help="Comma-separated ticker list"),
) -> None:
    """Ingest OHLCV from yfinance."""
    tlist = [t.strip().upper() for t in tickers.split(",")] if tickers else None
    stats = ingest_prices_flow(tickers=tlist, start=start, end=end)
    typer.echo(stats.model_dump_json(indent=2))


@app.command()
def macro(
    start: str | None = typer.Option(None, help="YYYY-MM-DD"),
    series: str | None = typer.Option(None, help="Comma-separated FRED series IDs"),
) -> None:
    """Ingest macro series from FRED."""
    slist = [s.strip().upper() for s in series.split(",")] if series else None
    stats = ingest_macro_flow(series=slist, start=start)
    typer.echo(stats.model_dump_json(indent=2))


@app.command()
def all(
    start: str | None = typer.Option(None, help="YYYY-MM-DD"),
) -> None:
    """Run prices + macro ingestion sequentially."""
    p = ingest_prices_flow(start=start)
    typer.echo(p.model_dump_json(indent=2))
    m = ingest_macro_flow(start=start)
    typer.echo(m.model_dump_json(indent=2))


if __name__ == "__main__":
    app()
