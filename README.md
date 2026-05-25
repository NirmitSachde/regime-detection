# Adaptive Market Regime Detection & Backtesting Platform

End-to-end pipeline that classifies the current market regime each day and
routes systematic trading strategies through regime-aware risk rules.
Fully reproducible from a clean clone, runs locally, **free data only**.

| | |
|---|---|
| **Live dashboard** | <https://nirmitsachde.github.io/regime-detection/> |
| **Live API** | <https://regime-detection-api.onrender.com/docs> (Swagger) |
| **Code reference** | <https://nirmitsachde.github.io/regime-detection/reference/> |
| **Source** | <https://github.com/NirmitSachde/regime-detection> |

[![CI](https://github.com/NirmitSachde/regime-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/NirmitSachde/regime-detection/actions/workflows/ci.yml)
[![Pages](https://github.com/NirmitSachde/regime-detection/actions/workflows/pages.yml/badge.svg)](https://github.com/NirmitSachde/regime-detection/actions/workflows/pages.yml)
[![Nightly refresh](https://github.com/NirmitSachde/regime-detection/actions/workflows/refresh.yml/badge.svg)](https://github.com/NirmitSachde/regime-detection/actions/workflows/refresh.yml)
[![python](https://img.shields.io/badge/python-3.12-blue)](pyproject.toml)
[![ruff](https://img.shields.io/badge/lint-ruff-orange)](pyproject.toml)
[![mypy](https://img.shields.io/badge/types-mypy_strict-blue)](pyproject.toml)
[![tests](https://img.shields.io/badge/tests-81%20passing-brightgreen)](tests/)
[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/NirmitSachde/regime-detection)

## Results — real backtest, 2018-01-02 to 2026-05-22 on SPY

| Strategy | Sharpe | CAGR | Max DD | Calmar | Note |
|---|---:|---:|---:|---:|---|
| Buy & hold | 0.55 | +12.8% | -33.7% | 0.38 | passive benchmark |
| Trend (50/200 MA, unconditional) | 0.51 | +7.0% | -35.7% | 0.19 | textbook trend rule |
| **Trend + regime overlay** | **0.65** | **+10.4%** | **-26.9%** | **0.39** | data-driven multipliers |
| Mean-rev (enabled in high-vol bear only) | 0.25 | +1.7% | -11.0% | 0.15 | small but positive |

**Headline:** the regime overlay improves Sharpe by 27% and Calmar by 105% vs the
unconditional trend rule. Max drawdown shrinks by 8.8 percentage points.

The per-regime trend Sharpes that drive the multipliers are themselves the
finding: the unconditional trend rule works in 3 out of 4 HMM states
(Sharpe 1.54, 1.38, 0.57) and loses money in the 4th (Sharpe -0.74). Sizing
1.5x in the first three and going flat in the fourth is the entire alpha.

## Stack

| Layer | Tool |
|---|---|
| Package management | **uv** |
| DataFrames | **Polars** (+ pandas interop) |
| Storage | **DuckDB + Parquet** (Hive-partitioned) |
| Orchestration | **Prefect 3** |
| Transformation | **dbt-core** + `dbt-duckdb` |
| ML | **hmmlearn** (Gaussian HMM) + **LightGBM** (supervised classifier) |
| Tracking | **MLflow** (local file store + model registry) |
| Backtesting | **vectorbt** |
| App | **Streamlit** (multi-page) + **Plotly** |
| Quality | **Ruff** + **mypy --strict** + **pytest --cov ≥ 80%** |
| CI | GitHub Actions |

## Architecture

```
yfinance + FRED ──► Prefect flows ──► Hive-partitioned Parquet (data/raw/)
                                              │
                                              ▼
                                       DuckDB warehouse
                                              │
                                              ▼
                          dbt: staging → intermediate → marts
                                              │
                                              ▼
                                  mart_features  /  mart_macro_features
                                              │
                  ┌───────────────────────────┼───────────────────────────┐
                  ▼                           ▼                           ▼
            hmmlearn HMM             LightGBM classifier            vectorbt
            (unsupervised)           (supervised, walk-fwd)         (backtest)
                  │                           │                           │
                  └───────────► MLflow ◄──────┘                           ▼
                                                                  Streamlit dashboard
                                                                          │
                                                                          ▼
                                                              ntfy.sh drift alerts
```

## Quick start

```bash
# 1. Install + bootstrap (uv handles Python, venv, deps)
make setup

# 2. Bring up Streamlit + MLflow + Prefect containers
make up
# → Streamlit:  http://localhost:8501
# → MLflow:     http://localhost:5000
# → Prefect:    http://localhost:4200

# 3. Run the pipeline (needs FRED_API_KEY in .env — free key at https://fredaccount.stlouisfed.org/apikeys)
make ingest && make dbt-build && make train && make backtest
```

## Layout

```
regime-detection/
├── src/regime/
│   ├── config.py             pydantic-settings, single source of config
│   ├── logging.py            structlog + rich
│   ├── ingestion/            Prefect flows + Pydantic contracts
│   ├── transform/            Polars feature + target builders
│   ├── models/               HMM, LightGBM, MLflow registry helpers
│   ├── backtest/             cost model, strategies, vectorbt engine
│   ├── monitoring/           drift check + ntfy webhook
│   ├── app/                  Streamlit multi-page app
│   ├── warehouse.py          DuckDB external-table bootstrap
│   └── cli.py                top-level Typer CLI
├── dbt/
│   ├── dbt_project.yml
│   ├── profiles.yml
│   └── models/
│       ├── staging/ stg_prices, stg_macro
│       ├── intermediate/ int_returns, int_technicals, int_realized_vol, int_macro_features
│       └── marts/ mart_features, mart_macro_features
├── tests/                    unit + integration; look-ahead-bias guard
├── notebooks/01_eda.ipynb    EDA + 12 chart exports
├── docs/                     research memo, ADRs, architecture, chart JSONs
├── scripts/run_pipeline.py   end-to-end orchestrator
├── data/                     (gitignored) raw Parquet, warehouse, mlruns
├── Dockerfile                multi-stage, uv-based
├── docker-compose.yml        app + mlflow + prefect
└── Makefile                  every workflow as a one-liner
```

## Quality gates

Every PR must pass:

```bash
make lint        # ruff check + ruff format --check
make typecheck   # mypy --strict src/
make test        # pytest with --cov-fail-under=80
make dbt-build   # full dbt build (sampled in CI)
```

The look-ahead-bias guard in [`tests/unit/test_look_ahead_bias.py`](tests/unit/test_look_ahead_bias.py)
empirically asserts that perturbing future data leaves day-`t` features unchanged.

## What's interesting

- **Idempotent ingestion** — content-hashed Parquet partitions; re-runs produce zero new bytes if data is unchanged.
- **dbt + Polars dual feature pipeline** — analytics-facing vs training-facing, with rationale in [ADR-0001](docs/adr/0001-dbt-vs-polars-redundancy.md).
- **BIC-selected HMM** — picks K from {3, 4} candidates by Bayesian Information Criterion.
- **Walk-forward CV** — `TimeSeriesSplit` with early stopping; never random splits.
- **Realistic cost model** — base + size-impact slippage, configurable borrow drag on shorts.
- **Bootstrap Sharpe CIs** — 1000 resamples, 95% confidence interval reported in every backtest summary.

## Constraints (hard)

- Free data only (yfinance + FRED)
- `mypy --strict` clean on all of `src/`
- ≥80% line coverage on core modules
- Reproducible from clean clone with `make setup && make up`

See [`01_handoff_regime_detection.md`](01_handoff_regime_detection.md) and
[`01_proposal_regime_detection.md`](01_proposal_regime_detection.md) for the
full executable spec and proposal (both local-only, gitignored).
