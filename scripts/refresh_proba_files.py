"""Regenerate data/models/hmm/proba.parquet and data/models/lgbm/proba.parquet
from the most recent trained models and the current warehouse.

Called by:
  - the nightly refresh GitHub Action (.github/workflows/refresh.yml)
  - manually whenever you re-train and want fresh probability artefacts
    without re-running the whole pipeline

Pipeline assumed to have run already:
  data/warehouse.duckdb           — populated dbt warehouse
  data/models/hmm/fit.pkl         — pickled HMMFitResult
  data/models/lgbm/model.txt      — saved LightGBM booster
  data/models/lgbm/metadata.json  — { feature_columns, classes, ... }
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import polars as pl

from regime.models.hmm_regime import predict_state_proba
from regime.transform.features import build_hmm_features, build_supervised_features

REPO_ROOT = Path(__file__).resolve().parent.parent


def regenerate_hmm_proba(data_dir: Path) -> None:
    fit_path = data_dir / "models" / "hmm" / "fit.pkl"
    if not fit_path.exists():
        print(f"skip HMM proba: {fit_path} not found", file=sys.stderr)
        return
    with open(fit_path, "rb") as fh:
        fit = pickle.load(fh)

    con = duckdb.connect(str(data_dir / "warehouse.duckdb"), read_only=True)
    try:
        macro = con.execute(
            "select * from main_marts.mart_macro_features order by feature_date"
        ).pl()
    finally:
        con.close()

    feats = build_hmm_features(macro)
    proba = predict_state_proba(fit, feats)
    out = pl.DataFrame(
        {
            "feature_date": feats["feature_date"],
            **{f"p{i}": proba[:, i].tolist() for i in range(fit.n_states)},
        }
    )
    out_path = data_dir / "models" / "hmm" / "proba.parquet"
    out.write_parquet(out_path)
    print(f"wrote {out_path}  ({out.height} rows, {fit.n_states} states)")


def regenerate_lgbm_proba(data_dir: Path) -> None:
    model_path = data_dir / "models" / "lgbm" / "model.txt"
    meta_path = data_dir / "models" / "lgbm" / "metadata.json"
    if not (model_path.exists() and meta_path.exists()):
        print("skip LGBM proba: model or metadata not found", file=sys.stderr)
        return

    booster = lgb.Booster(model_file=str(model_path))
    meta = json.loads(meta_path.read_text())
    feature_cols = meta["feature_columns"]
    classes = meta["classes"]

    con = duckdb.connect(str(data_dir / "warehouse.duckdb"), read_only=True)
    try:
        mart = con.execute(
            "select * from main_marts.mart_features where ticker='SPY' order by trade_date"
        ).pl()
    finally:
        con.close()

    feats = build_supervised_features(mart).drop_nulls(subset=feature_cols)
    x = feats.select(feature_cols).to_numpy().astype(np.float32)
    proba = np.asarray(booster.predict(x), dtype=np.float64)

    out = pl.DataFrame(
        {
            "feature_date": feats["trade_date"],
            "ticker": feats["ticker"],
            **{f"p{c}": proba[:, i].tolist() for i, c in enumerate(classes)},
        }
    )
    out_path = data_dir / "models" / "lgbm" / "proba.parquet"
    out.write_parquet(out_path)
    print(f"wrote {out_path}  ({out.height} rows)")


def main() -> None:
    data_dir = REPO_ROOT / "data"
    regenerate_hmm_proba(data_dir)
    regenerate_lgbm_proba(data_dir)


if __name__ == "__main__":
    main()
