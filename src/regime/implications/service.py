"""Assemble RegimeImplications payloads — the function the Streamlit page + API call.

Lookup order:
    1. Try the warehouse (data/warehouse.duckdb) + HMM labels parquet.
    2. If either is missing, fall back to synthetic data baked into the API
       sample module + synthetic regime statistics from policy.

Either way, the returned shape is identical. The `data_source` field on
the response tells the caller which path was taken.
"""

from __future__ import annotations

from datetime import date as _date
from pathlib import Path
from typing import TYPE_CHECKING

from regime.config import get_settings
from regime.implications import stats as stats_mod

if TYPE_CHECKING:
    import polars as pl
from regime.implications.models import (
    AlternativeScenario,
    RegimeImplications,
)
from regime.implications.policy import (
    ALLOCATION_BY_REGIME,
    REGIME_DESCRIPTIONS,
    REGIME_LABELS,
    RISK_PROFILE,
    caveats_for,
    confidence_label,
    headline_for_regime,
)


def _build_response(
    *,
    as_of: _date,
    regime: int,
    probabilities: dict[int, float],
    days_in_run: int | None,
    historical: stats_mod.HistoricalRegimeStats,
    data_source: str,
) -> RegimeImplications:
    confidence = float(probabilities.get(regime, max(probabilities.values(), default=1.0)))
    conf_label = confidence_label(confidence)

    alt: AlternativeScenario | None = None
    if conf_label != "High":
        # Pick the next-most-likely regime ≠ current
        ranked = sorted(probabilities.items(), key=lambda kv: kv[1], reverse=True)
        for r, p in ranked:
            if r != regime and p >= 0.15:
                alt = AlternativeScenario(
                    regime=r,
                    regime_label=REGIME_LABELS.get(r, "Unknown"),
                    probability=round(p, 4),
                    risk_profile=RISK_PROFILE.get(r, "Unknown"),
                    headline_change=_alt_headline(regime, r),
                )
                break

    return RegimeImplications(
        as_of=as_of,
        regime=regime,
        regime_label=REGIME_LABELS.get(regime, "Unknown"),
        risk_profile=RISK_PROFILE.get(regime, "Unknown"),
        description=REGIME_DESCRIPTIONS.get(regime, ""),
        confidence=round(confidence, 4),
        confidence_label=conf_label,
        probabilities={k: round(v, 4) for k, v in probabilities.items()},
        days_in_current_run=days_in_run,
        historical=historical,
        allocation=ALLOCATION_BY_REGIME.get(regime, []),
        headline=headline_for_regime(regime, confidence),
        alternative=alt,
        caveats=caveats_for(regime, confidence),
        data_source=data_source,
    )


def _alt_headline(current: int, alternative: int) -> str:
    """Plain-English description of what flips if we move from current → alternative."""
    pairs = {
        (0, 1): "Reduce equity overweight to neutral, take vol carry off, raise small cash buffer.",
        (0, 2): "Cut equity decisively, flip duration from underweight to overweight, add vol hedge.",
        (1, 0): "Lean equity overweight, sell the cash buffer, harvest the vol hedge.",
        (1, 2): "Cut equity and credit, extend duration, raise cash and vol-hedge sleeves.",
        (2, 0): "Add equity back aggressively, cut duration, sell the vol hedge.",
        (2, 1): "Restore equity toward benchmark, reduce duration overweight, taper the vol hedge.",
    }
    return pairs.get(
        (current, alternative),
        "Tilts shift toward the alternative regime's allocation.",
    )


def get_latest_implications() -> RegimeImplications:
    """Most-recent implications. Live if warehouse populated, synthetic otherwise."""
    settings = get_settings()
    labels_path = settings.data_dir / "models" / "hmm" / "labels.parquet"
    duckdb_path = settings.duckdb_path

    loaded = stats_mod.load_labels_and_returns(labels_path, duckdb_path)
    if loaded is not None:
        labels, returns = loaded
        if labels.height > 0:
            return _from_live(labels, returns)

    return _from_synthetic()


def get_implications_for_date(target: _date) -> RegimeImplications | None:
    """Implications for a specific date. None if no label is available."""
    settings = get_settings()
    labels_path = settings.data_dir / "models" / "hmm" / "labels.parquet"
    duckdb_path = settings.duckdb_path

    loaded = stats_mod.load_labels_and_returns(labels_path, duckdb_path)
    if loaded is None:
        return _synthetic_for_date(target)

    labels, returns = loaded
    if labels.height == 0:
        return _synthetic_for_date(target)

    import polars as pl  # noqa: PLC0415 - lazy import; live path only

    row = labels.filter(pl.col("feature_date") == target)
    if row.height == 0:
        return None

    regime = int(row["regime_state"][0])
    # Probability vector from labels alone is unobservable post-hoc; reconstruct a
    # one-hot with mild smoothing so confidence is still meaningful.
    probs = _reconstruct_probs(regime)
    days_in_run = _days_in_run_to(labels, target)
    hist = stats_mod.historical_stats_for_regime(labels, returns, regime)
    return _build_response(
        as_of=target,
        regime=regime,
        probabilities=probs,
        days_in_run=days_in_run,
        historical=hist,
        data_source="live",
    )


# ---------- Live + synthetic paths ----------


def _from_live(labels: "pl.DataFrame", returns: "pl.DataFrame") -> RegimeImplications:
    sorted_lbls = labels.sort("feature_date")
    latest_row = sorted_lbls.tail(1).to_dicts()[0]
    regime = int(latest_row["regime_state"])
    as_of = latest_row["feature_date"]
    probs = _reconstruct_probs(regime)
    days_in_run = stats_mod.days_in_current_run(sorted_lbls)
    hist = stats_mod.historical_stats_for_regime(sorted_lbls, returns, regime)
    return _build_response(
        as_of=as_of,
        regime=regime,
        probabilities=probs,
        days_in_run=days_in_run,
        historical=hist,
        data_source="live",
    )


def _from_synthetic() -> RegimeImplications:
    # Use the same baked-in sample data the FastAPI uses, for consistency
    from regime.api import sample as _sample

    latest = _sample.latest_regime()
    regime = int(latest["regime"])  # type: ignore[arg-type]
    as_of = _date.fromisoformat(latest["date"])  # type: ignore[arg-type]
    probs = {int(k): float(v) for k, v in latest["probabilities"].items()}  # type: ignore[union-attr]
    hist = stats_mod.synthetic_stats(regime)
    days_in_run = _synthetic_days_in_run(regime)
    return _build_response(
        as_of=as_of,
        regime=regime,
        probabilities=probs,
        days_in_run=days_in_run,
        historical=hist,
        data_source="synthetic",
    )


def _synthetic_for_date(target: _date) -> RegimeImplications | None:
    from regime.api import sample as _sample

    rec = _sample.regime_for_date(target)
    if rec is None:
        return None
    regime = int(rec["regime"])  # type: ignore[arg-type]
    probs = {int(k): float(v) for k, v in rec["probabilities"].items()}  # type: ignore[union-attr]
    hist = stats_mod.synthetic_stats(regime)
    return _build_response(
        as_of=target,
        regime=regime,
        probabilities=probs,
        days_in_run=None,
        historical=hist,
        data_source="synthetic",
    )


def _reconstruct_probs(regime: int) -> dict[int, float]:
    """Post-hoc proxy for HMM `predict_proba` when only the argmax label is stored."""
    base = {0: 0.05, 1: 0.05, 2: 0.05}
    base[regime] = 0.85
    if regime == 0:
        base[1] += 0.05
    elif regime == 1:
        base[0] += 0.025
        base[2] += 0.025
    else:
        base[1] += 0.05
    total = sum(base.values())
    return {k: v / total for k, v in base.items()}


def _days_in_run_to(labels: "pl.DataFrame", target: _date) -> int | None:
    import polars as pl  # noqa: PLC0415 - lazy import; live path only

    sub = labels.sort("feature_date").filter(pl.col("feature_date") <= target)
    if sub.height == 0:
        return None
    states = sub["regime_state"].to_list()
    latest = states[-1]
    n = 0
    for s in reversed(states):
        if s == latest:
            n += 1
        else:
            break
    return n


def _synthetic_days_in_run(regime: int) -> int:
    """Hand-tuned run length per regime, used for the synthetic-data path."""
    return {0: 38, 1: 12, 2: 21}.get(regime, 10)
