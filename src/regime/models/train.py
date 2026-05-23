"""Training CLI: `uv run regime-train hmm | lgbm | all`. All runs logged to MLflow."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import mlflow
import polars as pl
import typer

from regime.config import get_settings
from regime.logging import get_logger
from regime.models.hmm_regime import (
    HMMFitResult,
    fit_hmm,
    predict_states,
    summarize_states,
)
from regime.models.lightgbm_regime import LGBMFitResult, walk_forward_train
from regime.models.registry import log_dict, register_model, start_run
from regime.transform.features import build_hmm_features, build_supervised_features
from regime.transform.targets import join_labels_onto_features, label_from_hmm_states
from regime.warehouse import connect

log = get_logger(__name__)
app = typer.Typer(no_args_is_help=False)


def _load_macro() -> pl.DataFrame:
    con = connect()
    try:
        return con.execute("select * from main_marts.mart_macro_features").pl()
    finally:
        con.close()


def _load_mart() -> pl.DataFrame:
    con = connect()
    try:
        return con.execute("select * from main_marts.mart_features").pl()
    finally:
        con.close()


def _save_pickle(obj: object, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(obj, fh)
    return path


@app.command()
def hmm() -> HMMFitResult:
    """Fit Gaussian HMM on the macro panel and log to MLflow."""
    settings = get_settings()
    macro = _load_macro()
    feats = build_hmm_features(macro)
    if feats.height == 0:
        raise typer.BadParameter("No macro features available — run ingestion + dbt-build first")

    with start_run("hmm-fit", tags={"phase": "5"}):
        fit = fit_hmm(feats, n_states_candidates=(3, 4))
        log_dict(
            metrics={
                "bic": fit.bic,
                "log_likelihood": fit.log_likelihood,
                "n_states": float(fit.n_states),
                "n_features": float(len(fit.feature_columns)),
                "n_obs": float(feats.height),
            },
            params={
                "feature_columns": ",".join(fit.feature_columns),
                "covariance_type": "diag",
                "seed": settings.random_seed,
            },
        )
        states = predict_states(fit, feats)
        summary = summarize_states(fit, feats)
        log.info("hmm.state_summary", **{f"s{r['state']}": r["n_obs"] for r in summary.to_dicts()})

        # Persist labelled dates for the LGBM step
        labels = label_from_hmm_states(feats["feature_date"], states)
        out_dir = settings.data_dir / "models" / "hmm"
        out_dir.mkdir(parents=True, exist_ok=True)
        labels.write_parquet(out_dir / "labels.parquet")
        _save_pickle(fit, out_dir / "fit.pkl")

        mlflow.log_artifact(str(out_dir / "fit.pkl"))
        register_model(str(out_dir), name="regime-hmm")
    return fit


@app.command()
def lgbm() -> LGBMFitResult:
    """Train LightGBM classifier with walk-forward CV; log all folds to MLflow."""
    settings = get_settings()
    mart = _load_mart()
    feats = build_supervised_features(mart)

    labels_path = settings.data_dir / "models" / "hmm" / "labels.parquet"
    if not labels_path.exists():
        raise typer.BadParameter(
            "Run `regime-train hmm` first to produce regime labels."
        )
    labels = pl.read_parquet(labels_path)

    joined = join_labels_onto_features(feats, labels).drop_nulls(subset=["regime_state"])

    # Pick numeric, non-key columns as features
    key_cols = {"ticker", "trade_date", "regime_state"}
    feat_cols = [
        c
        for c in joined.columns
        if c not in key_cols and joined[c].dtype.is_numeric()
    ]

    with start_run("lgbm-fit", tags={"phase": "5"}):
        fit = walk_forward_train(joined, feat_cols, label_col="regime_state", n_splits=5)
        log_dict(
            metrics={
                "overall_macro_f1": fit.overall_macro_f1,
                "n_features": float(len(fit.feature_columns)),
                "n_classes": float(len(fit.classes)),
            },
            params={
                "feature_columns": ",".join(fit.feature_columns),
                "n_splits": "5",
                "seed": settings.random_seed,
            },
        )
        for f in fit.folds:
            mlflow.log_metric("fold_macro_f1", f.macro_f1, step=f.fold)

        out_dir = settings.data_dir / "models" / "lgbm"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "feature_columns": list(fit.feature_columns),
                    "classes": list(fit.classes),
                    "folds": [
                        {
                            "fold": f.fold,
                            "macro_f1": f.macro_f1,
                            "train_size": f.train_size,
                            "val_size": f.val_size,
                            "classes_seen": f.classes_seen,
                        }
                        for f in fit.folds
                    ],
                    "overall_macro_f1": fit.overall_macro_f1,
                },
                indent=2,
            )
        )
        fit.booster.save_model(str(out_dir / "model.txt"))
        mlflow.log_artifact(str(out_dir / "model.txt"))
        mlflow.log_artifact(str(out_dir / "metadata.json"))
        register_model(str(out_dir), name="regime-lgbm")
    return fit


@app.command()
def all() -> None:
    """Run hmm then lgbm."""
    hmm()
    lgbm()


# Module-level aliases used by scripts/run_pipeline.py
train_hmm = hmm
train_lgbm = lgbm


if __name__ == "__main__":
    app()
