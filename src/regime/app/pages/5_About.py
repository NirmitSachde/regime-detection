"""About page — links to MLflow, repo, research memo."""

from __future__ import annotations

import streamlit as st

from regime import __version__
from regime.config import get_settings

st.set_page_config(page_title="About", page_icon=":information_source:", layout="wide")
st.title("About")

settings = get_settings()

st.markdown(
    f"""
    **Version:** v{__version__}

    A production-grade pipeline for adaptive market regime detection and
    backtesting. Stack: Prefect 3 · DuckDB · dbt-core · hmmlearn · LightGBM
    · MLflow · vectorbt · Streamlit · Polars.

    ### Links
    - **MLflow UI** — [`{settings.mlflow_tracking_uri}`]({settings.mlflow_tracking_uri})
    - **Repo** — local clone
    - **Research memo** — `docs/research_memo.md`
    - **Architecture diagram** — `docs/architecture.png`

    ### Reproducibility
    ```bash
    make setup            # uv sync + pre-commit
    make up               # docker-compose: Streamlit + MLflow + Prefect
    make ingest           # yfinance + FRED → Parquet
    make dbt-build        # staging → intermediate → marts
    make train            # HMM + LightGBM, logged to MLflow
    make backtest         # vectorbt with bootstrap Sharpe CIs
    ```

    All data ingested from free sources (yfinance, FRED). No paid APIs or services.
    """
)
