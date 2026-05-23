"""Research page — embedded EDA charts + methodology notes."""

from __future__ import annotations

from pathlib import Path

import plotly.io as pio
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[3]
CHARTS_DIR = REPO_ROOT / "docs" / "charts"

st.set_page_config(page_title="Research", page_icon=":scroll:", layout="wide")
st.title("Research")

st.markdown(
    """
    Methodology, EDA charts, and notes on what worked / didn't.
    Full memo: `docs/research_memo.md`.
    """
)

if not CHARTS_DIR.exists():
    st.info("Run the EDA notebook to populate `docs/charts/`.")
else:
    charts = sorted(CHARTS_DIR.glob("*.json"))
    if not charts:
        st.info("No charts found in `docs/charts/`. Run `notebooks/01_eda.ipynb`.")
    for p in charts:
        st.subheader(p.stem)
        fig = pio.read_json(p)
        st.plotly_chart(fig, use_container_width=True)
