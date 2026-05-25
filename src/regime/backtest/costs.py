"""Vectorized cost model: commissions, linear slippage in size/ADV, borrow cost.

Designed to plug straight into vectorbt's `Portfolio.from_signals(fees=..., slippage=...)`
or — for more nuanced costs — to be pre-applied to a return series.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True, slots=True)
class CostConfig:
    commission_per_share: float = 0.0  # Alpaca paper default
    slippage_bps_base: float = 1.0  # 1 bp on liquid names
    slippage_size_coef: float = 50.0  # add bps proportional to size/ADV
    borrow_annual_rate: float = 0.005  # 50 bps on short positions
    short_only_borrow: bool = True


def slippage_bps(order_notional: pl.Series, adv_notional: pl.Series, cfg: CostConfig) -> pl.Series:
    """Slippage in basis points: base + size-impact term.

    impact = base + coef * (order / ADV)
    """
    impact = cfg.slippage_bps_base + cfg.slippage_size_coef * (
        order_notional / adv_notional.fill_null(strategy="forward")
    )
    return impact.clip(lower_bound=0.0, upper_bound=200.0)


def borrow_drag(positions: pl.Series, cfg: CostConfig) -> pl.Series:
    """Daily return drag from carrying a short position."""
    daily = cfg.borrow_annual_rate / TRADING_DAYS_PER_YEAR
    if cfg.short_only_borrow:
        return (-positions.clip(upper_bound=0.0).abs() * daily).alias("borrow_drag")
    return (-positions.abs() * daily).alias("borrow_drag")


_DEFAULT_COST_CFG = CostConfig()


def apply_costs_to_returns(
    strategy_returns: np.ndarray,
    positions: np.ndarray,
    cfg: CostConfig | None = None,
) -> np.ndarray:
    """Subtract per-day cost drag from a pre-computed strategy return series."""
    if cfg is None:
        cfg = _DEFAULT_COST_CFG
    if strategy_returns.shape != positions.shape:
        raise ValueError("strategy_returns and positions must have the same shape")

    turnover = np.abs(np.diff(positions, prepend=0.0))
    slip_bps = cfg.slippage_bps_base + cfg.slippage_size_coef * (turnover / 1.0)
    slip = (slip_bps / 1e4) * turnover

    borrow_daily = cfg.borrow_annual_rate / TRADING_DAYS_PER_YEAR
    borrow = (
        -np.clip(positions, a_min=None, a_max=0.0) * borrow_daily
        if cfg.short_only_borrow
        else -np.abs(positions) * borrow_daily
    )

    return np.asarray(strategy_returns - slip - borrow)
