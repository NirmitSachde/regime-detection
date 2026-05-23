"""Streamlit dashboard — multi-page entry point.

Pages live under `src/regime/app/pages/` and are auto-discovered by Streamlit.
"""

from __future__ import annotations

import streamlit as st

from regime import __version__
from regime.config import get_settings

st.set_page_config(
    page_title="Adaptive Market Regime Detection",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main() -> None:
    settings = get_settings()

    st.title("Adaptive Market Regime Detection & Backtesting")
    st.caption(f"v{__version__} — DuckDB · dbt · MLflow · vectorbt · Streamlit")

    st.markdown(
        """
        Welcome. This dashboard is the human-facing surface for a fully reproducible
        market-regime detection pipeline. Use the sidebar to navigate.

        - **Live Regime** — current regime probabilities + last 90 days of states.
        - **Backtest Replay** — pick a strategy, see equity / drawdown / trades.
        - **Research** — EDA charts and methodology notes.
        - **About** — links to MLflow, repo, research memo.
        """
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Universe size", len(settings.universe_list))
    col2.metric("History start", settings.history_start)
    col3.metric("Data dir", str(settings.data_dir))

    st.info(
        "Run `make ingest && make dbt-build && make train && make backtest` to populate "
        "the warehouse and models. Then refresh this page."
    )


if __name__ == "__main__":
    main()
