"""Target / label construction for regime-supervised learning.

The HMM produces an unsupervised state sequence. We promote those states to
labels for a downstream supervised classifier (LightGBM) so that we can score
each day's regime probability vector with features from the wider feature mart.
"""

from __future__ import annotations

import numpy as np
import polars as pl


def label_from_hmm_states(feature_dates: pl.Series, hmm_states: np.ndarray) -> pl.DataFrame:
    """Bind HMM-predicted states back onto dates."""
    if len(feature_dates) != len(hmm_states):
        raise ValueError(f"length mismatch: dates={len(feature_dates)} states={len(hmm_states)}")
    return pl.DataFrame(
        {
            "feature_date": feature_dates,
            "regime_state": pl.Series(values=hmm_states.astype("int64"), dtype=pl.Int64),
        }
    )


def join_labels_onto_features(features: pl.DataFrame, labels: pl.DataFrame) -> pl.DataFrame:
    """Left-join the per-date regime label onto a (ticker, trade_date) feature frame."""
    if "trade_date" not in features.columns:
        raise ValueError("features must have a `trade_date` column")
    if "feature_date" not in labels.columns:
        raise ValueError("labels must have a `feature_date` column")
    return features.join(
        labels.rename({"feature_date": "trade_date"}),
        on="trade_date",
        how="left",
    )
