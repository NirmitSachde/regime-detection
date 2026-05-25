"""Derive regime-conditional position-size multipliers from history.

The trend-overlay strategy's position multiplier per regime was previously
hand-coded ({0: 1.5, 1: 0.7, 2: 0.0}). That's defensible but not principled.

This module derives them from data: for each regime, compute the in-sample
Sharpe ratio of the unconditional trend signal restricted to that regime,
then map the per-regime Sharpe to a position multiplier so we lean into
regimes where trend has worked and shrink (or zero) in regimes where it
has not.

Mapping rule (intentionally conservative):
    Sharpe >= 0.5   →  multiplier 1.5  (lever up)
    Sharpe >= 0.0   →  multiplier 1.0  (full size)
    Sharpe >= -0.5  →  multiplier 0.5  (half size)
    Sharpe <  -0.5  →  multiplier 0.0  (flat)

Saved to data/models/regime_multipliers.json. Backtest engine reads it.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import duckdb
import polars as pl

from regime.config import get_settings
from regime.logging import get_logger

log = get_logger(__name__)

TRADING_DAYS_PER_YEAR = 252


def _per_regime_trend_sharpe(
    labels: pl.DataFrame,
    returns: pl.DataFrame,
    fast: int = 50,
    slow: int = 200,
) -> dict[int, float]:
    """Sharpe of the (fast/slow MA crossover, long-only) strategy per regime."""
    df = (
        labels.rename({"feature_date": "trade_date"})
        .join(returns, on="trade_date", how="inner")
        .sort("trade_date")
    )
    if df.height < slow + 50:
        return {}

    # Build the unconditional trend signal
    close = df["adj_close"].to_list()
    fast_ma = [None] * len(close)
    slow_ma = [None] * len(close)
    for i in range(len(close)):
        if i >= fast - 1:
            fast_ma[i] = sum(close[i - fast + 1 : i + 1]) / fast
        if i >= slow - 1:
            slow_ma[i] = sum(close[i - slow + 1 : i + 1]) / slow

    positions = [
        1.0 if (f is not None and s is not None and f > s) else 0.0
        for f, s in zip(fast_ma, slow_ma, strict=True)
    ]
    rets = df["log_ret_1d"].to_list()
    states = df["regime_state"].to_list()

    # Trend returns (lagged position by 1 to trade on prior day's signal)
    strat_rets: list[tuple[int, float]] = []
    for i in range(1, len(rets)):
        if rets[i] is None or states[i] is None:
            continue
        pos = positions[i - 1]
        r = pos * float(rets[i])
        strat_rets.append((int(states[i]), r))

    out: dict[int, float] = {}
    for state in {s for s, _ in strat_rets}:
        in_regime = [r for s, r in strat_rets if s == state]
        if len(in_regime) < 20:
            continue
        mean = sum(in_regime) / len(in_regime)
        var = sum((r - mean) ** 2 for r in in_regime) / len(in_regime)
        sd = math.sqrt(var) if var > 0 else 0.0
        sharpe = mean / sd * math.sqrt(TRADING_DAYS_PER_YEAR) if sd > 0 else 0.0
        out[state] = round(sharpe, 3)
    return out


def _sharpe_to_multiplier(sharpe: float) -> float:
    """Conservative step mapping. Doesn't lever beyond 1.5x."""
    if sharpe >= 0.5:
        return 1.5
    if sharpe >= 0.0:
        return 1.0
    if sharpe >= -0.5:
        return 0.5
    return 0.0


def derive_and_save(labels_path: Path | None = None) -> dict[int, float]:
    """Compute regime multipliers from history and persist them.

    Returns the multiplier dict {state: multiplier} for caller convenience.
    """
    settings = get_settings()
    labels_path = labels_path or settings.data_dir / "models" / "hmm" / "labels.parquet"
    if not labels_path.exists():
        log.warning("regime_multipliers.no_labels", path=str(labels_path))
        return {}

    labels = pl.read_parquet(labels_path)
    con = duckdb.connect(str(settings.duckdb_path), read_only=True)
    try:
        returns = con.execute(
            """
            select trade_date, adj_close, log_ret_1d
            from main_marts.mart_features
            where ticker = 'SPY' and log_ret_1d is not null
            order by trade_date
            """
        ).pl()
    finally:
        con.close()

    sharpes = _per_regime_trend_sharpe(labels, returns)
    multipliers = {s: _sharpe_to_multiplier(sh) for s, sh in sharpes.items()}

    out_path = settings.data_dir / "models" / "regime_multipliers.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "in_regime_trend_sharpe": {str(k): v for k, v in sharpes.items()},
        "multipliers": {str(k): v for k, v in multipliers.items()},
        "mapping_rule": (
            "sharpe>=0.5 → 1.5, sharpe>=0 → 1.0, sharpe>=-0.5 → 0.5, sharpe<-0.5 → 0.0"
        ),
        "note": (
            "State 0 is the bull regime (highest mean SPY return) after the "
            "post-training sort_states_by_mean_return remap."
        ),
    }
    out_path.write_text(json.dumps(payload, indent=2))
    log.info(
        "regime_multipliers.saved",
        path=str(out_path),
        sharpes=sharpes,
        multipliers=multipliers,
    )
    return multipliers


def load_multipliers() -> dict[int, float]:
    """Load saved regime multipliers, or fall back to conservative defaults."""
    settings = get_settings()
    path = settings.data_dir / "models" / "regime_multipliers.json"
    if path.exists():
        d = json.loads(path.read_text())
        return {int(k): float(v) for k, v in d.get("multipliers", {}).items()}
    # Conservative defaults — under-weight unknowns rather than over-leverage
    return {0: 1.0, 1: 0.5, 2: 0.0, 3: 0.0}


def _per_regime_meanrev_sharpe(
    labels: pl.DataFrame,
    returns: pl.DataFrame,
    long_thresh: float = 0.1,
    short_thresh: float = 0.9,
) -> dict[int, float]:
    """Sharpe of a simple Bollinger %B mean-reversion strategy per regime.

    A poor man's diagnostic: we don't have bb_pctb_20 here for SPY directly,
    so we approximate via z-score of price vs its 20-day MA + 2σ band.
    """
    df = (
        labels.rename({"feature_date": "trade_date"})
        .join(returns, on="trade_date", how="inner")
        .sort("trade_date")
    )
    if df.height < 50:
        return {}
    close = df["adj_close"].to_list()
    # 20d MA + std
    rolling_mean: list[float | None] = [None] * len(close)
    rolling_std: list[float | None] = [None] * len(close)
    for i in range(len(close)):
        if i >= 19:
            window = close[i - 19 : i + 1]
            mu = sum(window) / 20
            rolling_mean[i] = mu
            var = sum((x - mu) ** 2 for x in window) / 20
            rolling_std[i] = math.sqrt(var) if var > 0 else None

    pos: list[float] = []
    for i in range(len(close)):
        mu, sd = rolling_mean[i], rolling_std[i]
        if mu is None or sd is None or sd == 0:
            pos.append(0.0)
            continue
        bb = (close[i] - (mu - 2 * sd)) / (4 * sd)
        if bb < long_thresh:
            pos.append(1.0)
        elif bb > short_thresh:
            pos.append(-1.0)
        else:
            pos.append(0.0)

    rets = df["log_ret_1d"].to_list()
    states = df["regime_state"].to_list()
    strat_rets: list[tuple[int, float]] = []
    for i in range(1, len(rets)):
        if rets[i] is None or states[i] is None:
            continue
        r = pos[i - 1] * float(rets[i])
        strat_rets.append((int(states[i]), r))

    out: dict[int, float] = {}
    for state in {s for s, _ in strat_rets}:
        in_regime = [r for s, r in strat_rets if s == state]
        if len(in_regime) < 20:
            continue
        mean = sum(in_regime) / len(in_regime)
        var = sum((r - mean) ** 2 for r in in_regime) / len(in_regime)
        sd = math.sqrt(var) if var > 0 else 0.0
        out[state] = round(mean / sd * math.sqrt(TRADING_DAYS_PER_YEAR) if sd > 0 else 0.0, 3)
    return out


def derive_meanrev_enabled_regimes() -> set[int]:
    """Identify regimes where mean-reversion has positive Sharpe historically.

    Returns the set of regime indices to enable mean-rev in. Empty set if
    mean-rev hasn't worked anywhere historically — caller should disable
    the strategy.
    """
    settings = get_settings()
    labels_path = settings.data_dir / "models" / "hmm" / "labels.parquet"
    if not labels_path.exists():
        return {1}  # conservative default: only neutral regime

    labels = pl.read_parquet(labels_path)
    con = duckdb.connect(str(settings.duckdb_path), read_only=True)
    try:
        returns = con.execute(
            """
            select trade_date, adj_close, log_ret_1d
            from main_marts.mart_features
            where ticker = 'SPY' and log_ret_1d is not null
            order by trade_date
            """
        ).pl()
    finally:
        con.close()

    sharpes = _per_regime_meanrev_sharpe(labels, returns)
    enabled = {s for s, sh in sharpes.items() if sh > 0.3}  # require meaningful edge
    # Persist for inspection
    out_path = settings.data_dir / "models" / "meanrev_regimes.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "in_regime_meanrev_sharpe": {str(k): v for k, v in sharpes.items()},
                "enabled_regimes": sorted(enabled),
                "threshold": 0.3,
            },
            indent=2,
        )
    )
    log.info("meanrev_regimes.derived", sharpes=sharpes, enabled=sorted(enabled))
    return enabled
