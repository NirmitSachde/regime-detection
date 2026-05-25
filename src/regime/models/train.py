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
    sort_states_by_mean_return,
    summarize_states,
)
from regime.models.lightgbm_regime import (
    LGBMFitResult,
    tune_hyperparameters,
    walk_forward_train,
)
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

        # Sort states deterministically: state 0 = highest mean SPY return
        # (bull), state K-1 = lowest (bear). Removes the arbitrary state
        # indexing the EM algorithm produces and makes labels stable across
        # training runs.
        if "spy_ret_21d" in feats.columns:
            fit, remap = sort_states_by_mean_return(fit, feats, feats["spy_ret_21d"])
            log.info("hmm.state_remap", remap={int(k): int(v) for k, v in remap.items()})

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

        # Also persist the full predict_proba matrix so the API can surface
        # the actual HMM posterior instead of a smoothed-argmax reconstruction.
        from regime.models.hmm_regime import predict_state_proba

        proba = predict_state_proba(fit, feats)
        proba_df = pl.DataFrame(
            {
                "feature_date": feats["feature_date"],
                **{f"p{i}": proba[:, i].tolist() for i in range(fit.n_states)},
            }
        )
        proba_df.write_parquet(out_dir / "proba.parquet")

        mlflow.log_artifact(str(out_dir / "fit.pkl"))
        register_model(str(out_dir), name="regime-hmm")
    return fit


@app.command()
def lgbm(
    tune: bool = typer.Option(False, help="Run Optuna hyperparameter search before training"),
    n_trials: int = typer.Option(50, help="Number of Optuna trials when --tune"),
) -> LGBMFitResult:
    """Train LightGBM classifier with walk-forward CV; log all folds to MLflow.

    Pass --tune to run an Optuna hyperparameter search first (TPE sampler,
    walk-forward macro-F1 objective). Best params are saved to
    data/models/lgbm/best_params.json and used for the final fit.
    """
    settings = get_settings()
    mart = _load_mart()
    feats = build_supervised_features(mart)

    labels_path = settings.data_dir / "models" / "hmm" / "labels.parquet"
    if not labels_path.exists():
        raise typer.BadParameter("Run `regime-train hmm` first to produce regime labels.")
    labels = pl.read_parquet(labels_path)

    joined = join_labels_onto_features(feats, labels).drop_nulls(subset=["regime_state"])

    # Pick numeric, non-key columns as features
    key_cols = {"ticker", "trade_date", "regime_state"}
    feat_cols = [c for c in joined.columns if c not in key_cols and joined[c].dtype.is_numeric()]

    best_params: dict[str, object] | None = None
    out_dir_early = settings.data_dir / "models" / "lgbm"
    out_dir_early.mkdir(parents=True, exist_ok=True)
    best_params_path = out_dir_early / "best_params.json"

    if tune:
        log.info("lgbm.tune.start", n_trials=n_trials)
        best = tune_hyperparameters(joined, feat_cols, n_trials=n_trials, n_splits=5)
        best_macro_f1 = float(best.pop("best_macro_f1"))  # type: ignore[arg-type]
        best_params = best
        # Persist the best params for reproducibility + future re-runs
        best_params_path.write_text(
            json.dumps(
                {"best_macro_f1": best_macro_f1, "params": best_params},
                indent=2,
            )
        )
        log.info("lgbm.tune.complete", best_macro_f1=round(best_macro_f1, 4))
    elif best_params_path.exists():
        # Re-use the previous best_params if available, so subsequent
        # untuned runs benefit from prior search work.
        prev = json.loads(best_params_path.read_text())
        best_params = prev.get("params")
        log.info(
            "lgbm.reuse_best_params",
            path=str(best_params_path),
            best_macro_f1=prev.get("best_macro_f1"),
        )

    # Build the params dict for walk_forward_train
    n_class = len({int(c) for c in joined["regime_state"].drop_nulls()})
    train_params: dict[str, object] | None
    if best_params:
        num_boost_round = int(best_params.pop("num_boost_round", 400))  # type: ignore[call-overload]
        train_params = {
            "objective": "multiclass",
            "num_class": n_class,
            "metric": "multi_logloss",
            "verbosity": -1,
            "seed": settings.random_seed,
            **best_params,
        }
    else:
        num_boost_round = 400
        train_params = None  # walk_forward_train will use its defaults

    with start_run("lgbm-fit", tags={"phase": "5", "tuned": str(tune)}):
        fit = walk_forward_train(
            joined,
            feat_cols,
            label_col="regime_state",
            n_splits=5,
            params=train_params,
            num_boost_round=num_boost_round,
        )
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
