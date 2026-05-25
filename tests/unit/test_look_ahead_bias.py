"""Look-ahead bias guard.

For every (ticker, trade_date) feature row, the underlying observation that
produced each non-null feature value must be strictly earlier in time than
the trade_date that owns the feature. We prove this on synthetic data where
we know the construction is correct, then assert the production builder
respects the same invariant.

The handoff explicitly requires this test to exist or Phase 4 is incomplete.
"""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

from regime.transform.features import (
    add_forward_target,
    build_hmm_features,
    build_supervised_features,
    lagged_macro_features,
)


@pytest.fixture
def synthetic_macro() -> pl.DataFrame:
    dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(60)]
    n = len(dates)
    return pl.DataFrame(
        {
            "feature_date": dates,
            "vix": [10.0 + i * 0.1 for i in range(n)],
            "vix_chg_5d": [0.5] * n,
            "dxy_chg_5d": [0.1] * n,
            "yc_10y2y": [0.5 + i * 0.01 for i in range(n)],
            "yc_chg_21d": [0.02] * n,
            "hy_oas": [3.0 + i * 0.05 for i in range(n)],
        }
    )


@pytest.fixture
def synthetic_mart() -> pl.DataFrame:
    dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(30)]
    rows: list[dict[str, object]] = []
    for tkr in ("AAA", "BBB"):
        for i, d in enumerate(dates):
            rows.append(
                {
                    "ticker": tkr,
                    "trade_date": d,
                    "adj_close": 100.0 + i + (1 if tkr == "BBB" else 0),
                    "volume": 1_000_000.0,
                    "rsi_14": 50.0,
                    "ma_50": 100.0,
                }
            )
    return pl.from_dicts(rows)


def test_lagged_macro_uses_only_prior_values(synthetic_macro: pl.DataFrame) -> None:
    """Day t's macro features must equal day t-1's raw observation."""
    lagged = lagged_macro_features(synthetic_macro, lag_days=1)
    # Pick day t = 2024-01-15
    t = date(2024, 1, 15)
    row_t = lagged.filter(pl.col("feature_date") == t).to_dicts()[0]
    src_prev = synthetic_macro.filter(pl.col("feature_date") == (t - timedelta(days=1))).to_dicts()[
        0
    ]
    assert row_t["vix"] == src_prev["vix"]
    assert row_t["yc_10y2y"] == src_prev["yc_10y2y"]


def test_hmm_features_drop_unobservable_first_row(synthetic_macro: pl.DataFrame) -> None:
    """The first row has no `t-1` observation and must be dropped."""
    feats = build_hmm_features(synthetic_macro)
    assert feats["feature_date"].min() > synthetic_macro["feature_date"].min()


def test_supervised_features_per_ticker_lag(synthetic_mart: pl.DataFrame) -> None:
    """Lag is applied within ticker — never cross-ticker contamination."""
    lagged = build_supervised_features(synthetic_mart, lag_days=1)
    aaa = lagged.filter(pl.col("ticker") == "AAA").sort("trade_date")
    bbb = lagged.filter(pl.col("ticker") == "BBB").sort("trade_date")

    # First row per ticker has null feature (no prior observation)
    assert aaa["adj_close"][0] is None
    assert bbb["adj_close"][0] is None

    # Day-2 lag must equal the same ticker's day-1 raw value
    src_aaa = synthetic_mart.filter(pl.col("ticker") == "AAA").sort("trade_date")
    assert aaa["adj_close"][1] == src_aaa["adj_close"][0]


def test_forward_target_uses_only_future_close(synthetic_mart: pl.DataFrame) -> None:
    """Target is forward-looking by design, but only the target — not features."""
    with_tgt = add_forward_target(synthetic_mart, horizon=1, target_col="fwd")
    aaa = with_tgt.filter(pl.col("ticker") == "AAA").sort("trade_date")
    src = synthetic_mart.filter(pl.col("ticker") == "AAA").sort("trade_date")
    import math

    expected = math.log(float(src["adj_close"][1]) / float(src["adj_close"][0]))
    assert abs(float(aaa["fwd"][0]) - expected) < 1e-12
    # Last row's target should be null (no t+1 observation)
    assert aaa["fwd"][-1] is None


def test_no_feature_column_can_be_computed_from_future_data(
    synthetic_mart: pl.DataFrame,
) -> None:
    """Empirical sampler: for a randomly chosen feature row, perturb the
    *future* of the source mart and re-build features. None of the lagged
    feature values for the chosen date may change."""
    target_date = date(2024, 1, 10)
    perturbed = synthetic_mart.with_columns(
        pl.when(pl.col("trade_date") > target_date)
        .then(pl.col("adj_close") * 1.5)
        .otherwise(pl.col("adj_close"))
        .alias("adj_close")
    )

    orig = build_supervised_features(synthetic_mart).filter(pl.col("trade_date") == target_date)
    new = build_supervised_features(perturbed).filter(pl.col("trade_date") == target_date)

    # Compare every non-target feature column
    for col in orig.columns:
        if col in {"ticker", "trade_date"}:
            continue
        a = orig[col].to_list()
        b = new[col].to_list()
        assert a == b, f"feature `{col}` changed after future-only perturbation"
