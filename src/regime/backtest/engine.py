"""Backtest engine: thin wrapper around vectorbt + our cost model.

Outputs a BacktestResult that's safe to persist and to render in Streamlit.
"""

from __future__ import annotations

import json
import uuid
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from regime.backtest.costs import TRADING_DAYS_PER_YEAR, CostConfig, apply_costs_to_returns
from regime.backtest.strategies import (
    StrategySignal,
    baseline_trend,
    regime_conditioned_meanrev,
    regime_conditioned_trend,
)
from regime.config import get_settings
from regime.logging import get_logger
from regime.warehouse import connect

log = get_logger(__name__)


class BacktestSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy: str
    ticker: str
    start: date
    end: date
    capital: float
    sharpe: float
    sortino: float
    cagr: float
    max_drawdown: float
    calmar: float
    hit_rate: float
    exposure: float
    turnover: float
    n_trades: int
    bootstrap_sharpe_ci_low: float
    bootstrap_sharpe_ci_high: float


class BacktestResult(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    run_id: str
    summary: BacktestSummary
    equity_curve_path: str = Field(description="Parquet with columns: date, equity")
    returns_path: str = Field(description="Parquet with columns: date, ret, position")


def _to_returns(prices: pl.DataFrame) -> pl.Series:
    return (
        prices.sort("trade_date")["adj_close"].log().diff()
    ).fill_null(0.0)


def _bootstrap_sharpe_ci(
    returns: np.ndarray, n: int = 1000, conf: float = 0.95, seed: int = 42
) -> tuple[float, float]:
    if returns.size < 50:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    sharpes = np.empty(n, dtype=np.float64)
    for i in range(n):
        sample = rng.choice(returns, size=returns.size, replace=True)
        std = sample.std()
        sharpes[i] = sample.mean() / std * np.sqrt(TRADING_DAYS_PER_YEAR) if std > 0 else 0.0
    lo, hi = np.percentile(sharpes, [(1 - conf) / 2 * 100, (1 + conf) / 2 * 100])
    return float(lo), float(hi)


def _stats(returns: np.ndarray, positions: np.ndarray) -> dict[str, float]:
    if returns.size == 0:
        return {k: 0.0 for k in (
            "sharpe", "sortino", "cagr", "max_drawdown", "calmar",
            "hit_rate", "exposure", "turnover", "n_trades",
        )}

    mu = returns.mean()
    sd = returns.std()
    down = returns[returns < 0]
    sd_down = down.std() if down.size else 0.0

    equity = (1.0 + returns).cumprod()
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    max_dd = float(dd.min())

    years = returns.size / TRADING_DAYS_PER_YEAR
    cagr = float(equity[-1] ** (1 / years) - 1) if years > 0 and equity[-1] > 0 else 0.0
    sharpe = float(mu / sd * np.sqrt(TRADING_DAYS_PER_YEAR)) if sd > 0 else 0.0
    sortino = float(mu / sd_down * np.sqrt(TRADING_DAYS_PER_YEAR)) if sd_down > 0 else 0.0
    calmar = float(cagr / abs(max_dd)) if max_dd != 0 else 0.0
    hit = float((returns > 0).sum() / returns.size)
    exposure = float((np.abs(positions) > 0).mean())
    trades = np.count_nonzero(np.diff(positions, prepend=0.0))
    turnover = float(np.abs(np.diff(positions, prepend=0.0)).sum())

    return {
        "sharpe": sharpe,
        "sortino": sortino,
        "cagr": cagr,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "hit_rate": hit,
        "exposure": exposure,
        "turnover": turnover,
        "n_trades": float(trades),
    }


def run_backtest(
    strategy_name: str,
    ticker: str,
    capital: float = 100_000.0,
    cost_cfg: CostConfig | None = None,
    regime_states: pl.DataFrame | None = None,
    regime_multipliers: dict[int, float] | None = None,
    enabled_regimes: set[int] | None = None,
) -> BacktestResult:
    settings = get_settings()
    cost_cfg = cost_cfg or CostConfig()

    con = connect()
    try:
        prices = con.execute(
            f"select * from main_marts.mart_features where ticker = '{ticker}' order by trade_date"
        ).pl()
    finally:
        con.close()
    if prices.height == 0:
        raise ValueError(f"no prices for {ticker}")

    if strategy_name == "baseline_trend":
        signal = baseline_trend(prices)
    elif strategy_name == "regime_conditioned_trend":
        if regime_states is None or regime_multipliers is None:
            raise ValueError("regime_conditioned_trend needs regime_states + regime_multipliers")
        signal = regime_conditioned_trend(prices, regime_states, regime_multipliers)
    elif strategy_name == "regime_conditioned_meanrev":
        if regime_states is None or enabled_regimes is None:
            raise ValueError("regime_conditioned_meanrev needs regime_states + enabled_regimes")
        signal = regime_conditioned_meanrev(prices, regime_states, enabled_regimes)
    else:
        raise ValueError(f"unknown strategy: {strategy_name}")

    returns = _to_returns(prices).to_numpy()
    # Position lagged by 1 day so we trade on day t using day t-1's signal
    positions = np.roll(signal.target_position, 1)
    positions[0] = 0.0
    strat_returns = returns * positions
    strat_returns = apply_costs_to_returns(strat_returns, positions, cost_cfg)

    metrics = _stats(strat_returns, positions)
    lo, hi = _bootstrap_sharpe_ci(strat_returns)

    run_id = uuid.uuid4().hex[:12]
    out_dir = settings.data_dir / "backtests" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    eq_df = pd.DataFrame(
        {
            "trade_date": prices.sort("trade_date")["trade_date"].to_pandas(),
            "equity": (1.0 + strat_returns).cumprod() * capital,
        }
    )
    eq_path = out_dir / "equity.parquet"
    eq_df.to_parquet(eq_path)

    ret_df = pd.DataFrame(
        {
            "trade_date": prices.sort("trade_date")["trade_date"].to_pandas(),
            "ret": strat_returns,
            "position": positions,
        }
    )
    ret_path = out_dir / "returns.parquet"
    ret_df.to_parquet(ret_path)

    summary = BacktestSummary(
        strategy=strategy_name,
        ticker=ticker,
        start=date.fromisoformat(str(prices["trade_date"].min())),
        end=date.fromisoformat(str(prices["trade_date"].max())),
        capital=capital,
        sharpe=metrics["sharpe"],
        sortino=metrics["sortino"],
        cagr=metrics["cagr"],
        max_drawdown=metrics["max_drawdown"],
        calmar=metrics["calmar"],
        hit_rate=metrics["hit_rate"],
        exposure=metrics["exposure"],
        turnover=metrics["turnover"],
        n_trades=int(metrics["n_trades"]),
        bootstrap_sharpe_ci_low=lo,
        bootstrap_sharpe_ci_high=hi,
    )
    (out_dir / "summary.json").write_text(summary.model_dump_json(indent=2))

    log.info("backtest.done", run_id=run_id, **summary.model_dump(mode="json"))
    return BacktestResult(
        run_id=run_id,
        summary=summary,
        equity_curve_path=str(eq_path),
        returns_path=str(ret_path),
    )


def run_all_backtests(ticker: str = "SPY") -> list[BacktestResult]:
    settings = get_settings()
    labels_path = settings.data_dir / "models" / "hmm" / "labels.parquet"

    results: list[BacktestResult] = []
    results.append(run_backtest("baseline_trend", ticker))

    if labels_path.exists():
        labels = pl.read_parquet(labels_path)
        # Heuristic multipliers — refine after inspecting state stats
        mults = {0: 0.5, 1: 1.0, 2: 1.5, 3: 0.0}
        results.append(
            run_backtest(
                "regime_conditioned_trend",
                ticker,
                regime_states=labels,
                regime_multipliers=mults,
            )
        )
        results.append(
            run_backtest(
                "regime_conditioned_meanrev",
                ticker,
                regime_states=labels,
                enabled_regimes={0, 1},
            )
        )

    # Persist a combined summary table for the Streamlit page
    out = pd.DataFrame([r.summary.model_dump(mode="json") for r in results])
    out_path = settings.data_dir / "backtests" / "summary_latest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps([r.summary.model_dump(mode="json") for r in results], indent=2))
    return results
