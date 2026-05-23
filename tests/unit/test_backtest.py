"""Cost model + signal-generation behavior."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from regime.backtest.costs import CostConfig, apply_costs_to_returns
from regime.backtest.strategies import baseline_trend, regime_conditioned_trend


def test_costs_no_trade_no_drag() -> None:
    returns = np.zeros(10)
    positions = np.zeros(10)
    out = apply_costs_to_returns(returns, positions, CostConfig())
    assert np.allclose(out, 0.0)


def test_costs_short_borrow_drag_is_negative() -> None:
    returns = np.zeros(252)
    positions = -np.ones(252)
    cfg = CostConfig(slippage_bps_base=0.0, slippage_size_coef=0.0)
    out = apply_costs_to_returns(returns, positions, cfg)
    # Holding short for 1 year at 50 bps should accumulate ~50 bps drag
    assert out.sum() < 0
    assert abs(out.sum() + cfg.borrow_annual_rate) < 1e-3


@pytest.fixture
def trending_prices() -> pl.DataFrame:
    dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(400)]
    close = np.linspace(100, 200, len(dates))
    return pl.DataFrame({"trade_date": dates, "adj_close": close.tolist()})


def test_baseline_trend_goes_long_in_uptrend(trending_prices: pl.DataFrame) -> None:
    sig = baseline_trend(trending_prices, fast=50, slow=200)
    # After the slow window warms up, position should be 1
    assert sig.target_position[-1] == 1.0


def test_baseline_trend_zero_before_warmup(trending_prices: pl.DataFrame) -> None:
    sig = baseline_trend(trending_prices, fast=50, slow=200)
    assert sig.target_position[0] == 0.0
    assert sig.target_position[100] == 0.0


def test_regime_conditioned_trend_zeroes_in_bear_state(trending_prices: pl.DataFrame) -> None:
    # Make every date a "bear" regime (state 0 -> multiplier 0)
    states = pl.DataFrame(
        {"feature_date": trending_prices["trade_date"], "regime_state": [0] * trending_prices.height}
    )
    sig = regime_conditioned_trend(
        trending_prices, states, regime_multipliers={0: 0.0, 1: 1.0}
    )
    assert sig.target_position[-1] == 0.0
