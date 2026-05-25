"""Smoke test: HMM fits, BIC selects a candidate, predictions are sane."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from regime.models.hmm_regime import fit_hmm, predict_state_proba, predict_states


@pytest.fixture
def two_regime_synthetic() -> pl.DataFrame:
    """Two clearly-separated Gaussian regimes — HMM should recover them easily."""
    rng = np.random.default_rng(42)
    n = 200
    a = rng.normal(loc=[0.0, 0.0], scale=[0.5, 0.5], size=(n, 2))
    b = rng.normal(loc=[3.0, 3.0], scale=[0.5, 0.5], size=(n, 2))
    x = np.vstack([a, b])
    dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(2 * n)]
    return pl.DataFrame({"feature_date": dates, "f0": x[:, 0].tolist(), "f1": x[:, 1].tolist()})


def test_fit_returns_valid_result(two_regime_synthetic: pl.DataFrame) -> None:
    fit = fit_hmm(two_regime_synthetic, feature_cols=["f0", "f1"], n_states_candidates=(2, 3))
    assert fit.n_states in {2, 3}
    assert fit.feature_columns == ("f0", "f1")
    assert np.isfinite(fit.bic)


def test_predict_states_length_matches(two_regime_synthetic: pl.DataFrame) -> None:
    fit = fit_hmm(two_regime_synthetic, feature_cols=["f0", "f1"], n_states_candidates=(2,))
    s = predict_states(fit, two_regime_synthetic)
    assert len(s) == two_regime_synthetic.height
    assert s.dtype == np.int64


def test_predict_proba_rows_sum_to_one(two_regime_synthetic: pl.DataFrame) -> None:
    fit = fit_hmm(two_regime_synthetic, feature_cols=["f0", "f1"], n_states_candidates=(2,))
    p = predict_state_proba(fit, two_regime_synthetic)
    sums = p.sum(axis=1)
    assert np.allclose(sums, 1.0, atol=1e-6)
