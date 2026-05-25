"""Backtest Replay page — pick a strategy, see equity / drawdown / trades."""

from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import streamlit as st

from regime.config import get_settings

st.set_page_config(page_title="Backtest Replay", page_icon=":bar_chart:", layout="wide")
st.title("Backtest Replay")

settings = get_settings()
summary_path = settings.data_dir / "backtests" / "summary_latest.json"

if not summary_path.exists():
    st.warning("No backtests yet. Run `make backtest`.")
    st.stop()

summaries = json.loads(summary_path.read_text())
labels = [f"{s['strategy']} / {s['ticker']}" for s in summaries]
choice = st.selectbox(
    "Strategy", options=list(range(len(summaries))), format_func=lambda i: labels[i]
)
sm = summaries[choice]

st.subheader("Summary")
cols = st.columns(4)
cols[0].metric(
    "Sharpe",
    f"{sm['sharpe']:.2f}",
    f"[{sm['bootstrap_sharpe_ci_low']:.2f}, {sm['bootstrap_sharpe_ci_high']:.2f}]",
)
cols[1].metric("CAGR", f"{sm['cagr'] * 100:.2f}%")
cols[2].metric("Max DD", f"{sm['max_drawdown'] * 100:.2f}%")
cols[3].metric("Calmar", f"{sm['calmar']:.2f}")

cols2 = st.columns(4)
cols2[0].metric("Sortino", f"{sm['sortino']:.2f}")
cols2[1].metric("Hit rate", f"{sm['hit_rate'] * 100:.1f}%")
cols2[2].metric("Exposure", f"{sm['exposure'] * 100:.1f}%")
cols2[3].metric("# trades", sm["n_trades"])

# Find the latest run dir matching strategy+ticker
runs_root = settings.data_dir / "backtests"
matching = []
for d in runs_root.iterdir():
    if not d.is_dir():
        continue
    sj = d / "summary.json"
    if not sj.exists():
        continue
    s = json.loads(sj.read_text())
    if s.get("strategy") == sm["strategy"] and s.get("ticker") == sm["ticker"]:
        matching.append((d, s))

if not matching:
    st.info("Equity curve files not found.")
    st.stop()

matching.sort(key=lambda kv: kv[0].stat().st_mtime, reverse=True)
latest_dir = matching[0][0]
eq = pd.read_parquet(latest_dir / "equity.parquet")
fig = px.line(eq, x="trade_date", y="equity", title="Equity curve")
st.plotly_chart(fig, use_container_width=True)

eq["peak"] = eq["equity"].cummax()
eq["dd"] = (eq["equity"] - eq["peak"]) / eq["peak"]
fig2 = px.area(eq, x="trade_date", y="dd", title="Drawdown")
st.plotly_chart(fig2, use_container_width=True)
