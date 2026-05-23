"""In-memory Polars feature builders.

These mirror the dbt logic but operate on Polars frames for training-time
feature construction (HMM input on the macro panel, LightGBM input from the
mart). The dbt models remain the single source of truth for analytics-facing
queries; this module is the source of truth for model training.

ADR-001 documents the deliberate redundancy.

Every feature in this module is built so that the value for date `t` uses
only data observable strictly before the close of `t-1`. This is enforced by
the look-ahead-bias guard test in `tests/unit/test_look_ahead_bias.py`.
"""

from __future__ import annotations

import math

import polars as pl


def log_returns(close: pl.Series, horizon: int = 1) -> pl.Series:
    return (close.log() - close.shift(horizon).log()).alias(f"log_ret_{horizon}d")


def realized_vol(returns: pl.Series, window: int = 21, ann: int = 252) -> pl.Series:
    return (returns.rolling_std(window) * math.sqrt(ann)).alias(f"rv_{window}d")


def lagged_macro_features(macro: pl.DataFrame, lag_days: int = 1) -> pl.DataFrame:
    """Lag every numeric macro column by `lag_days` to avoid look-ahead.

    The macro dbt mart already represents observation date; lagging here
    ensures that day-t models never see the day-t print.
    """
    if "feature_date" not in macro.columns:
        raise ValueError("macro must contain a `feature_date` column")
    numeric = [c for c in macro.columns if c != "feature_date" and macro[c].dtype.is_numeric()]
    return macro.sort("feature_date").with_columns(
        [pl.col(c).shift(lag_days).alias(c) for c in numeric]
    )


def build_hmm_features(macro: pl.DataFrame) -> pl.DataFrame:
    """HMM input: realized-vol-like, change-in-DXY, yield curve slope.

    Returns a frame with `feature_date` and feature columns; rows with any
    null feature are dropped (HMM cannot consume NaN).
    """
    lagged = lagged_macro_features(macro, lag_days=1)
    feat_cols = [
        c for c in ("vix", "vix_chg_5d", "dxy_chg_5d", "yc_10y2y", "yc_chg_21d", "hy_oas")
        if c in lagged.columns
    ]
    return lagged.select(["feature_date", *feat_cols]).drop_nulls()


def build_supervised_features(
    mart_features: pl.DataFrame, lag_days: int = 1
) -> pl.DataFrame:
    """LightGBM input: lag every non-key column to avoid look-ahead."""
    keys = {"ticker", "trade_date"}
    feat_cols = [c for c in mart_features.columns if c not in keys]
    df = mart_features.sort(["ticker", "trade_date"])
    return df.with_columns(
        [pl.col(c).shift(lag_days).over("ticker").alias(c) for c in feat_cols]
    )


def add_forward_target(
    df: pl.DataFrame, horizon: int = 1, target_col: str = "fwd_return"
) -> pl.DataFrame:
    """Add forward log return as the supervised target.

    Forward returns are the only column built using future information, and
    they MUST be the target, not a feature. The bias guard test enforces this.
    """
    if "adj_close" not in df.columns:
        raise ValueError("add_forward_target requires `adj_close`")
    return df.sort(["ticker", "trade_date"]).with_columns(
        (
            pl.col("adj_close").shift(-horizon).over("ticker").log()
            - pl.col("adj_close").log()
        ).alias(target_col)
    )
