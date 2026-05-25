"""Implications module: policy completeness, confidence labels, service contract."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from regime.implications import (
    ALLOCATION_BY_REGIME,
    RegimeImplications,
    get_implications_for_date,
    get_latest_implications,
)
from regime.implications.policy import (
    REGIME_DESCRIPTIONS,
    RISK_PROFILE,
    caveats_for,
    confidence_label,
    headline_for_regime,
)
from regime.implications.service import (
    _alt_headline,
    _build_response,
    _reconstruct_probs,
)
from regime.implications.stats import (
    _episode_runs,
    _max_drawdown,
    days_in_current_run,
    historical_stats_for_regime,
    synthetic_stats,
)


# ---------- Policy: completeness + sanity ----------


@pytest.mark.parametrize("regime", [0, 1, 2])
def test_every_regime_has_a_full_allocation(regime: int) -> None:
    tilts = ALLOCATION_BY_REGIME[regime]
    classes = {t.asset_class for t in tilts}
    assert classes == {"Equity", "Duration", "Credit", "Cash", "Vol hedge"}


@pytest.mark.parametrize("regime", [0, 1, 2])
def test_every_regime_has_description_and_risk_profile(regime: int) -> None:
    assert REGIME_DESCRIPTIONS[regime]
    assert RISK_PROFILE[regime] in {"Risk-On", "Neutral", "Risk-Off"}


def test_bull_regime_overweights_equity() -> None:
    tilts = {t.asset_class: t for t in ALLOCATION_BY_REGIME[0]}
    assert tilts["Equity"].tilt == "Overweight"
    assert tilts["Equity"].bps > 0
    assert tilts["Cash"].tilt == "Underweight"


def test_bear_regime_underweights_equity_overweights_duration() -> None:
    tilts = {t.asset_class: t for t in ALLOCATION_BY_REGIME[2]}
    assert tilts["Equity"].tilt == "Underweight"
    assert tilts["Equity"].bps < 0
    assert tilts["Duration"].tilt == "Overweight"
    assert tilts["Duration"].bps > 0
    assert tilts["Vol hedge"].tilt == "Overweight"


def test_neutral_regime_stays_near_benchmark() -> None:
    tilts = {t.asset_class: t for t in ALLOCATION_BY_REGIME[1]}
    # Equity at benchmark
    assert tilts["Equity"].bps == 0
    # All magnitudes small in absolute terms
    assert all(abs(t.bps) <= 300 for t in ALLOCATION_BY_REGIME[1])


# ---------- Confidence labels ----------


@pytest.mark.parametrize(
    ("p", "expected"),
    [
        (0.95, "High"),
        (0.80, "High"),
        (0.75, "High"),
        (0.74, "Medium"),
        (0.60, "Medium"),
        (0.50, "Medium"),
        (0.49, "Low"),
        (0.34, "Low"),
    ],
)
def test_confidence_label_boundaries(p: float, expected: str) -> None:
    assert confidence_label(p) == expected


# ---------- Caveats ----------


def test_low_confidence_adds_caveat() -> None:
    high = caveats_for(regime=0, confidence=0.95)
    low = caveats_for(regime=0, confidence=0.40)
    assert len(low) > len(high)


def test_bear_regime_adds_implementation_caveat() -> None:
    cavs = caveats_for(regime=2, confidence=0.95)
    assert any("implementation costs" in c.lower() for c in cavs)


# ---------- Stats helpers ----------


def test_episode_runs_groups_correctly() -> None:
    assert _episode_runs([0, 0, 1, 1, 1, 0]) == [(0, 2), (1, 3), (0, 1)]
    assert _episode_runs([2]) == [(2, 1)]
    assert _episode_runs([]) == []


def test_max_drawdown_negative_or_zero() -> None:
    assert _max_drawdown([]) == 0.0
    assert _max_drawdown([0.01, 0.01, 0.01]) == 0.0
    # Down 10%, then up — DD captures the trough
    dd = _max_drawdown([-0.05, -0.05, 0.05])
    assert dd < 0
    assert dd >= -0.15


def test_days_in_current_run_counts_back() -> None:
    df = pl.DataFrame(
        {
            "feature_date": [date(2024, 1, i) for i in range(1, 8)],
            "regime_state": [0, 0, 1, 1, 2, 2, 2],
        }
    )
    assert days_in_current_run(df) == 3


def test_days_in_current_run_handles_empty() -> None:
    df = pl.DataFrame(schema={"feature_date": pl.Date, "regime_state": pl.Int64})
    assert days_in_current_run(df) is None


def test_historical_stats_uses_log_returns_when_present() -> None:
    labels = pl.DataFrame(
        {
            "feature_date": [date(2024, 1, i) for i in range(1, 11)],
            "regime_state": [0] * 10,
        }
    )
    returns = pl.DataFrame(
        {
            "trade_date": [date(2024, 1, i) for i in range(1, 11)],
            "log_ret_1d": [0.001, 0.002, -0.001, 0.003, 0.0, 0.001, 0.002, -0.002, 0.001, 0.002],
        }
    )
    stats = historical_stats_for_regime(labels, returns, regime=0)
    assert stats.n_episodes == 1
    assert stats.total_days == 10
    assert stats.sample_basis.startswith("SPY")
    assert 0 <= stats.hit_rate_pct <= 100


def test_historical_stats_handles_empty_input() -> None:
    empty = pl.DataFrame(schema={"feature_date": pl.Date, "regime_state": pl.Int64})
    stats = historical_stats_for_regime(empty, empty, regime=0)
    assert stats.n_episodes == 0


def test_synthetic_stats_match_expected_shape() -> None:
    s = synthetic_stats(2)
    assert s.annualized_return_pct < 0
    assert s.max_drawdown_pct < 0
    assert s.annualized_vol_pct > 30


# ---------- Service: shape + invariants ----------


def test_get_latest_returns_full_payload() -> None:
    imp = get_latest_implications()
    assert isinstance(imp, RegimeImplications)
    assert imp.regime in {0, 1, 2}
    assert imp.risk_profile in {"Risk-On", "Neutral", "Risk-Off"}
    assert imp.confidence_label in {"High", "Medium", "Low"}
    assert 0.0 <= imp.confidence <= 1.0
    assert len(imp.allocation) == 5
    assert imp.data_source in {"live", "synthetic"}
    assert imp.headline
    assert imp.caveats


def test_get_for_specific_date_synthetic_returns_payload() -> None:
    # The synthetic data covers 2018-01-02 onward; pick something inside the range.
    imp = get_implications_for_date(date(2020, 6, 15))
    assert imp is not None
    assert imp.regime in {0, 1, 2}


def test_get_for_unknown_date_returns_none() -> None:
    assert get_implications_for_date(date(1900, 1, 1)) is None


def test_probabilities_sum_close_to_one() -> None:
    imp = get_latest_implications()
    total = sum(imp.probabilities.values())
    assert 0.95 < total < 1.05


def test_reconstruct_probs_normalised() -> None:
    for r in (0, 1, 2):
        probs = _reconstruct_probs(r)
        assert abs(sum(probs.values()) - 1.0) < 1e-9
        # Argmax matches the input
        assert max(probs, key=lambda k: probs[k]) == r


# ---------- Alternative scenario ----------


def test_alternative_scenario_fires_when_confidence_low() -> None:
    response = _build_response(
        as_of=date(2024, 1, 10),
        regime=0,
        probabilities={0: 0.45, 1: 0.30, 2: 0.25},
        days_in_run=5,
        historical=synthetic_stats(0),
        data_source="synthetic",
    )
    assert response.alternative is not None
    assert response.alternative.regime != 0


def test_alternative_scenario_suppressed_when_high_confidence() -> None:
    response = _build_response(
        as_of=date(2024, 1, 10),
        regime=0,
        probabilities={0: 0.92, 1: 0.05, 2: 0.03},
        days_in_run=5,
        historical=synthetic_stats(0),
        data_source="synthetic",
    )
    assert response.alternative is None


def test_alt_headlines_cover_all_transitions() -> None:
    for cur in (0, 1, 2):
        for alt in (0, 1, 2):
            if cur == alt:
                continue
            assert _alt_headline(cur, alt)  # non-empty string


# ---------- Headline string ----------


def test_headline_mentions_risk_profile_and_confidence() -> None:
    s = headline_for_regime(2, 0.81)
    assert "Risk-Off" in s
    assert "81%" in s


# ---------- API integration ----------


def test_api_exposes_implications_latest() -> None:
    from fastapi.testclient import TestClient

    from regime.api.main import app

    client = TestClient(app)
    r = client.get("/regime/implications/latest")
    assert r.status_code == 200
    body = r.json()
    assert body["regime"] in (0, 1, 2)
    assert len(body["allocation"]) == 5
    assert "headline" in body


def test_api_implications_for_date_404_on_unknown() -> None:
    from fastapi.testclient import TestClient

    from regime.api.main import app

    client = TestClient(app)
    r = client.get("/regime/implications/1900-01-01")
    assert r.status_code == 404
