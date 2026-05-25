"""Live Regime page — current regime probabilities + last 90 days of states."""

from __future__ import annotations

import plotly.express as px
import polars as pl
import streamlit as st

from regime.config import get_settings

st.set_page_config(page_title="Live Regime", page_icon=":radio_button:", layout="wide")
st.title("Live Regime")

settings = get_settings()
labels_path = settings.data_dir / "models" / "hmm" / "labels.parquet"

if not labels_path.exists():
    st.warning("HMM labels not found. Run `make ingest dbt-build train`.")
    st.stop()

labels = pl.read_parquet(labels_path).sort("feature_date")
recent = labels.tail(90)

col1, col2 = st.columns([1, 2])
with col1:
    latest = labels.tail(1).to_dicts()[0]
    st.metric("Latest state", int(latest["regime_state"]))
    st.metric("As of", str(latest["feature_date"]))
    st.metric("Total observations", labels.height)

with col2:
    fig = px.scatter(
        recent.to_pandas(),
        x="feature_date",
        y="regime_state",
        color="regime_state",
        title="Regime states — last 90 days",
        labels={"feature_date": "Date", "regime_state": "State"},
    )
    fig.update_traces(marker={"size": 10})
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Per-state distribution")
counts = labels.group_by("regime_state").len().sort("regime_state")
st.dataframe(counts.to_pandas(), use_container_width=True, hide_index=True)
