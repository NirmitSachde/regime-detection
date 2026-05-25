"""Gaussian HMM wrapper for unsupervised market-regime discovery."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

from regime.config import get_settings
from regime.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class HMMFitResult:
    model: GaussianHMM
    scaler: StandardScaler
    feature_columns: tuple[str, ...]
    n_states: int
    bic: float
    log_likelihood: float


def _bic(model: GaussianHMM, x: np.ndarray) -> float:
    n_obs = x.shape[0]
    n_features = x.shape[1]
    n_params = (
        model.n_components
        - 1  # initial probs
        + model.n_components * (model.n_components - 1)  # transitions
        + model.n_components * n_features  # means
        + model.n_components * n_features  # diag covs
    )
    ll = float(model.score(x))
    return float(-2.0 * ll + n_params * np.log(n_obs))


def fit_hmm(
    features: pl.DataFrame,
    feature_cols: list[str] | None = None,
    n_states_candidates: tuple[int, ...] = (3, 4),
    n_iter: int = 200,
    seed: int | None = None,
) -> HMMFitResult:
    """Fit Gaussian HMMs for each K and pick by BIC."""
    settings = get_settings()
    seed = seed if seed is not None else settings.random_seed
    cols = feature_cols or [c for c in features.columns if c != "feature_date"]

    arr = features.select(cols).to_numpy().astype(np.float64)
    scaler = StandardScaler().fit(arr)
    x = scaler.transform(arr)

    best: HMMFitResult | None = None
    for k in n_states_candidates:
        model = GaussianHMM(
            n_components=k,
            covariance_type="diag",
            n_iter=n_iter,
            random_state=seed,
            tol=1e-3,
        )
        model.fit(x)
        bic = _bic(model, x)
        ll = float(model.score(x))
        log.info("hmm.candidate", k=k, bic=round(bic, 2), log_likelihood=round(ll, 2))
        cand = HMMFitResult(
            model=model,
            scaler=scaler,
            feature_columns=tuple(cols),
            n_states=k,
            bic=bic,
            log_likelihood=ll,
        )
        if best is None or bic < best.bic:
            best = cand
    assert best is not None
    log.info("hmm.selected", k=best.n_states, bic=round(best.bic, 2))
    return best


def predict_states(fit: HMMFitResult, features: pl.DataFrame) -> np.ndarray:
    x = fit.scaler.transform(features.select(list(fit.feature_columns)).to_numpy())
    return np.asarray(fit.model.predict(x), dtype=np.int64)


def predict_state_proba(fit: HMMFitResult, features: pl.DataFrame) -> np.ndarray:
    x = fit.scaler.transform(features.select(list(fit.feature_columns)).to_numpy())
    return np.asarray(fit.model.predict_proba(x), dtype=np.float64)


def sort_states_by_mean_return(
    fit: HMMFitResult,
    features: pl.DataFrame,
    spy_returns: pl.Series,
) -> tuple[HMMFitResult, dict[int, int]]:
    """Remap HMM state indices so state 0 = highest-mean-return ("Bull"),
    state K-1 = lowest-mean-return ("Bear"). Returns the remapped fit and
    the {old_state: new_state} dictionary.

    HMMs assign state indices arbitrarily; without this, "state 0" might be
    bear in one training run and bull in the next. Sorting by mean SPY return
    gives a stable, interpretable label ordering that matches human intuition
    (bull = positive returns, bear = negative).
    """
    states = predict_states(fit, features)
    if len(states) != len(spy_returns):
        # Mismatched lengths: skip remapping rather than error
        return fit, {i: i for i in range(fit.n_states)}

    ret_arr = np.array([float(r) if r is not None else 0.0 for r in spy_returns])
    # Mean return per state (ignore NaN, missing → 0)
    mean_by_state: dict[int, float] = {}
    for s in range(fit.n_states):
        mask = states == s
        if mask.any():
            mean_by_state[s] = float(np.nanmean(ret_arr[mask]))
        else:
            mean_by_state[s] = 0.0

    # Sort old states by descending mean return → new index 0 = bull, K-1 = bear
    old_order = sorted(mean_by_state.keys(), key=lambda s: mean_by_state[s], reverse=True)
    remap = {old: new for new, old in enumerate(old_order)}

    # Remap by permuting the model's internal parameters
    perm = np.array(old_order)
    new_model = GaussianHMM(
        n_components=fit.n_states,
        covariance_type="diag",
        n_iter=fit.model.n_iter,
        random_state=fit.model.random_state,
        tol=fit.model.tol,
    )
    # Copy-permute fitted parameters
    new_model.startprob_ = fit.model.startprob_[perm]
    new_model.transmat_ = fit.model.transmat_[perm][:, perm]
    new_model.means_ = fit.model.means_[perm]
    new_model.covars_ = (
        fit.model._covars_[perm] if hasattr(fit.model, "_covars_") else fit.model.covars_[perm]
    )
    new_model.n_features = fit.model.n_features

    new_fit = HMMFitResult(
        model=new_model,
        scaler=fit.scaler,
        feature_columns=fit.feature_columns,
        n_states=fit.n_states,
        bic=fit.bic,
        log_likelihood=fit.log_likelihood,
    )
    log.info(
        "hmm.states_sorted_by_return",
        remap={int(k): int(v) for k, v in remap.items()},
        mean_returns={int(k): round(v, 5) for k, v in mean_by_state.items()},
    )
    return new_fit, remap


def summarize_states(
    fit: HMMFitResult,
    features: pl.DataFrame,
    returns: pl.Series | None = None,
) -> pl.DataFrame:
    """Per-state diagnostics: persistence, count, mean return when provided."""
    states = predict_states(fit, features)
    df = pl.DataFrame({"state": states})
    if returns is not None and len(returns) == len(states):
        df = df.with_columns(returns.alias("ret"))
        return (
            df.group_by("state")
            .agg(
                pl.len().alias("n_obs"),
                pl.col("ret").mean().alias("mean_ret"),
                pl.col("ret").std().alias("std_ret"),
            )
            .sort("state")
        )
    return df.group_by("state").agg(pl.len().alias("n_obs")).sort("state")
