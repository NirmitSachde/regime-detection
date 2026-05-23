"""Backtest CLI."""

from __future__ import annotations

import typer

from regime.backtest.engine import run_all_backtests, run_backtest

app = typer.Typer(no_args_is_help=False)


@app.command()
def one(
    strategy: str = typer.Argument(...),
    ticker: str = typer.Option("SPY"),
    capital: float = typer.Option(100_000.0),
) -> None:
    """Run a single strategy/ticker backtest."""
    res = run_backtest(strategy, ticker, capital=capital)
    typer.echo(res.summary.model_dump_json(indent=2))


@app.command()
def all(ticker: str = typer.Option("SPY")) -> None:
    """Run all built-in strategies on `ticker`."""
    results = run_all_backtests(ticker)
    for r in results:
        typer.echo(r.summary.model_dump_json(indent=2))


if __name__ == "__main__":
    app()
