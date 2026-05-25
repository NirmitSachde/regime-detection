"""LightGBM multi-class classifier over HMM-derived regime labels.

The labelling phase fits the HMM once on the macro panel; this module then
trains a supervised model with the wider feature mart so we can score
per-ticker daily regimes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import TimeSeriesSplit

from regime.config import get_settings
from regime.logging import get_logger

log = get_logger(__name__)


@dataclass(slots=True)
class FoldMetric:
    fold: int
    train_size: int
    val_size: int
    macro_f1: float
    classes_seen: list[int]


@dataclass(slots=True)
class LGBMFitResult:
    booster: lgb.Booster
    feature_columns: tuple[str, ...]
    classes: tuple[int, ...]
    folds: list[FoldMetric] = field(default_factory=list)
    overall_macro_f1: float = 0.0


def _to_xy(
    df: pl.DataFrame, feature_cols: list[str], label_col: str
) -> tuple[np.ndarray, np.ndarray]:
    sub = df.drop_nulls(subset=[*feature_cols, label_col])
    x = sub.select(feature_cols).to_numpy().astype(np.float32)
    y = sub.select(label_col).to_numpy().ravel().astype(np.int64)
    return x, y


def walk_forward_train(
    df: pl.DataFrame,
    feature_cols: list[str],
    label_col: str = "regime_state",
    n_splits: int = 5,
    params: dict[str, object] | None = None,
    num_boost_round: int = 400,
) -> LGBMFitResult:
    """Expanding-window walk-forward CV training."""
    settings = get_settings()
    df = df.sort(["trade_date", "ticker"])
    x, y = _to_xy(df, feature_cols, label_col)
    if x.shape[0] == 0:
        raise ValueError("no rows after dropna; check feature/label columns")

    classes = sorted(int(c) for c in np.unique(y))
    n_class = len(classes)

    params = params or {
        "objective": "multiclass",
        "num_class": n_class,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 5,
        "metric": "multi_logloss",
        "verbosity": -1,
        "seed": settings.random_seed,
    }

    tss = TimeSeriesSplit(n_splits=n_splits)
    folds: list[FoldMetric] = []
    last_booster: lgb.Booster | None = None

    for i, (tr, va) in enumerate(tss.split(x)):
        dtr = lgb.Dataset(x[tr], label=y[tr])
        dva = lgb.Dataset(x[va], label=y[va], reference=dtr)
        booster = lgb.train(
            params,
            dtr,
            num_boost_round=num_boost_round,
            valid_sets=[dva],
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )
        pred = np.asarray(booster.predict(x[va])).argmax(axis=1)
        f1 = float(f1_score(y[va], pred, average="macro", zero_division=0))
        seen = sorted(int(c) for c in np.unique(y[va]))
        log.info("lgbm.fold", fold=i, macro_f1=round(f1, 4), n_train=len(tr), n_val=len(va))
        folds.append(
            FoldMetric(fold=i, train_size=len(tr), val_size=len(va), macro_f1=f1, classes_seen=seen)
        )
        last_booster = booster

    # Final model: refit on all data with the best iteration count seen
    final_iter = (
        max(
            (b.best_iteration for b in [last_booster] if b and b.best_iteration),
            default=num_boost_round,
        )
        or num_boost_round
    )
    dall = lgb.Dataset(x, label=y)
    final = lgb.train(params, dall, num_boost_round=final_iter)

    overall = float(np.mean([f.macro_f1 for f in folds])) if folds else 0.0
    log.info("lgbm.training.done", folds=len(folds), overall_macro_f1=round(overall, 4))
    return LGBMFitResult(
        booster=final,
        feature_columns=tuple(feature_cols),
        classes=tuple(classes),
        folds=folds,
        overall_macro_f1=overall,
    )


def predict_proba(fit: LGBMFitResult, df: pl.DataFrame) -> np.ndarray:
    x = df.select(list(fit.feature_columns)).to_numpy().astype(np.float32)
    return np.asarray(fit.booster.predict(x), dtype=np.float64)


def classification_report_text(fit: LGBMFitResult, df: pl.DataFrame, label_col: str) -> str:
    x, y = _to_xy(df, list(fit.feature_columns), label_col)
    pred = np.asarray(fit.booster.predict(x)).argmax(axis=1)
    return str(classification_report(y, pred, zero_division=0))
