"""Signal generators. Pure numpy / polars — no broker calls."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import polars as pl


@dataclass(frozen=True, slots=True)
class StrategySignal:
    """Aligned signal frame: dates, target_position in [-1, 1]."""

    dates: pl.Series
    target_position: np.ndarray  # shape (T,), values in [-1, 1]


def baseline_trend(prices: pl.DataFrame, fast: int = 50, slow: int = 200) -> StrategySignal:
    """Strategy A: long when fast MA > slow MA, flat otherwise."""
    p = prices.sort("trade_date")
    close = p["adj_close"].to_numpy().astype(np.float64)
    if close.shape[0] <= slow:
        return StrategySignal(dates=p["trade_date"], target_position=np.zeros(close.shape[0]))
    fast_ma = _rolling_mean(close, fast)
    slow_ma = _rolling_mean(close, slow)
    pos = np.where(fast_ma > slow_ma, 1.0, 0.0)
    pos[: slow - 1] = 0.0
    return StrategySignal(dates=p["trade_date"], target_position=pos)


def regime_conditioned_trend(
    prices: pl.DataFrame,
    regime_states: pl.DataFrame,
    regime_multipliers: dict[int, float],
    fast: int = 50,
    slow: int = 200,
) -> StrategySignal:
    """Strategy B: baseline trend with position size scaled by regime multiplier."""
    base = baseline_trend(prices, fast=fast, slow=slow)
    joined = (
        pl.DataFrame({"trade_date": base.dates, "base_pos": base.target_position})
        .join(regime_states.rename({"feature_date": "trade_date"}), on="trade_date", how="left")
    )
    states = joined["regime_state"].fill_null(strategy="forward").fill_null(0).to_numpy()
    mults = np.array(
        [regime_multipliers.get(int(s), 1.0) for s in states], dtype=np.float64
    )
    pos = joined["base_pos"].to_numpy() * mults
    return StrategySignal(dates=joined["trade_date"], target_position=pos)


def regime_conditioned_meanrev(
    prices_with_bbpb: pl.DataFrame,
    regime_states: pl.DataFrame,
    enabled_regimes: set[int],
    long_thresh: float = 0.1,
    short_thresh: float = 0.9,
) -> StrategySignal:
    """Strategy C: Bollinger %B mean reversion, only enabled in chosen regimes."""
    p = prices_with_bbpb.sort("trade_date")
    bb = p["bb_pctb_20"].to_numpy().astype(np.float64)
    raw_pos = np.where(bb < long_thresh, 1.0, np.where(bb > short_thresh, -1.0, 0.0))

    joined = (
        pl.DataFrame({"trade_date": p["trade_date"], "raw_pos": raw_pos})
        .join(regime_states.rename({"feature_date": "trade_date"}), on="trade_date", how="left")
    )
    states = joined["regime_state"].fill_null(strategy="forward").fill_null(-1).to_numpy()
    enabled = np.isin(states, list(enabled_regimes))
    pos = np.where(enabled, joined["raw_pos"].to_numpy(), 0.0)
    return StrategySignal(dates=joined["trade_date"], target_position=pos)


def _rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return x.copy()
    out = np.full_like(x, fill_value=np.nan, dtype=np.float64)
    cumsum = np.cumsum(np.insert(x, 0, 0.0))
    out[w - 1 :] = (cumsum[w:] - cumsum[:-w]) / w
    return out


# Registry — referenced by the CLI
STRATEGY_REGISTRY: dict[str, Callable[..., StrategySignal]] = {
    "baseline_trend": baseline_trend,
    "regime_conditioned_trend": regime_conditioned_trend,
    "regime_conditioned_meanrev": regime_conditioned_meanrev,
}
