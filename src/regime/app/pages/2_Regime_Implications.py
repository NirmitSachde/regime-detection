"""Regime Implications — translate the current regime into PM-facing guidance.

What a portfolio manager actually wants from a regime model: not the state
label, not the probability vector, but "given this, what should I do?"

This page renders a recommendation. The numbers come from
`regime.implications.get_latest_implications()`, which lives in a separate
module so the same payload can be served via the FastAPI surface.
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from regime.implications import get_latest_implications

st.set_page_config(
    page_title="Regime Implications",
    page_icon=":compass:",
    layout="wide",
)

st.title("Regime Implications")
st.caption(
    "What the current regime classification means for asset allocation — "
    "translated from probability vectors into PM-facing guidance."
)

implications = get_latest_implications()

if implications.data_source == "synthetic":
    st.info(
        "Showing synthetic-data implications because the warehouse is not yet "
        "populated. Run `make ingest && make dbt-build && make train` to switch "
        "this page to live model output."
    )

# ---------- Headline ----------
risk_color = {
    "Risk-On": "#4a9d6e",
    "Neutral": "#a8a29e",
    "Risk-Off": "#c14040",
}.get(implications.risk_profile, "#a8a29e")

st.markdown(
    f"""
    <div style="
        padding: 24px 28px;
        border: 1px solid #232323;
        border-left: 4px solid {risk_color};
        border-radius: 8px;
        background: #141414;
        margin: 12px 0 28px 0;
    ">
        <div style="font-family: 'JetBrains Mono', monospace;
                    font-size: 11px; color: #8a8a8a;
                    letter-spacing: .1em; text-transform: uppercase;
                    margin-bottom: 8px;">
            As of {implications.as_of}  ·  {implications.regime_label}
        </div>
        <div style="font-size: 22px; font-weight: 500; color: #ededed;
                    line-height: 1.4;">
            {implications.headline}
        </div>
        <div style="font-size: 14px; color: #a3a3a3; margin-top: 12px;
                    line-height: 1.6;">
            {implications.description}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- Confidence + run row ----------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Risk profile", implications.risk_profile)
col2.metric(
    "Model confidence",
    f"{implications.confidence * 100:.0f}%",
    implications.confidence_label,
)
col3.metric(
    "Days in current run",
    implications.days_in_current_run if implications.days_in_current_run else "—",
)
col4.metric(
    "Historical episode length",
    f"{implications.historical.avg_duration_days:.0f}d (avg)",
)

st.divider()

# ---------- Allocation tilt grid ----------
st.subheader("Suggested allocation tilt")
st.caption(
    "Deviations from a 60/30/10 (equity/duration/credit) benchmark. "
    "Cash and vol-hedge sleeves are taken from the equity bucket when invoked."
)

tilt_color = {
    "Overweight": "#4a9d6e",
    "Underweight": "#c14040",
    "Neutral": "#a8a29e",
}

cols = st.columns(len(implications.allocation))
for col, tilt in zip(cols, implications.allocation, strict=True):
    color = tilt_color.get(tilt.tilt, "#8a8a8a")
    sign = "+" if tilt.bps > 0 else ""
    col.markdown(
        f"""
        <div style="
            padding: 18px 16px;
            border: 1px solid #232323;
            border-radius: 6px;
            background: #141414;
            height: 100%;
        ">
            <div style="font-family: 'JetBrains Mono', monospace;
                        font-size: 10.5px; color: #8a8a8a;
                        text-transform: uppercase; letter-spacing: .12em;
                        margin-bottom: 12px;">
                {tilt.asset_class}
            </div>
            <div style="font-size: 13px; color: {color}; font-weight: 500;
                        margin-bottom: 4px;">
                {tilt.tilt}
            </div>
            <div style="font-family: 'JetBrains Mono', monospace;
                        font-size: 22px; color: {color}; font-weight: 500;
                        line-height: 1; margin-bottom: 8px;">
                {sign}{tilt.bps} bps
            </div>
            <div style="font-size: 11px; color: #5a5a5a; margin-bottom: 10px;
                        text-transform: uppercase; letter-spacing: .08em;">
                {tilt.magnitude}
            </div>
            <div style="font-size: 12.5px; color: #a3a3a3; line-height: 1.55;">
                {tilt.rationale}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()

# ---------- Two-column: historical stats + probability vector ----------
left, right = st.columns([3, 2])

with left:
    st.subheader(f"Historical performance — {implications.regime_label}")
    st.caption(
        f"What past instances of this regime looked like. "
        f"Computed on {implications.historical.sample_basis}."
    )

    h = implications.historical
    h_col1, h_col2, h_col3 = st.columns(3)
    h_col1.metric(
        "Annualised return",
        f"{h.annualized_return_pct:+.1f}%",
        help="Compounded mean daily return × 252",
    )
    h_col1.metric(
        "Hit rate",
        f"{h.hit_rate_pct:.0f}%",
        help="Percentage of days with positive return while in this regime",
    )
    h_col2.metric(
        "Annualised vol",
        f"{h.annualized_vol_pct:.1f}%",
        help="Sample stdev × √252",
    )
    h_col2.metric(
        "Max drawdown",
        f"{h.max_drawdown_pct:.1f}%",
        help="Worst peak-to-trough within episodes of this regime",
    )
    h_col3.metric(
        "Median daily return",
        f"{h.median_daily_return_pct:+.3f}%",
    )
    h_col3.metric(
        "Episodes",
        f"{h.n_episodes}",
        f"{h.total_days} total days",
    )

with right:
    st.subheader("Probability vector")
    st.caption("Where the model's mass is sitting today.")

    labels_short = {0: "Bull", 1: "Chop", 2: "Bear"}
    colors = {0: "#4a9d6e", 1: "#a8a29e", 2: "#c14040"}
    items = sorted(implications.probabilities.items())
    fig = go.Figure(
        go.Bar(
            x=[round(p * 100, 1) for _, p in items],
            y=[labels_short.get(int(k), str(k)) for k, _ in items],
            orientation="h",
            marker={"color": [colors.get(int(k), "#8a8a8a") for k, _ in items]},
            text=[f"{p * 100:.0f}%" for _, p in items],
            textposition="outside",
            hoverinfo="skip",
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#ededed", "family": "Inter"},
        margin={"l": 50, "r": 50, "t": 10, "b": 30},
        height=200,
        xaxis={
            "range": [0, 100],
            "showgrid": False,
            "ticksuffix": "%",
            "tickfont": {"color": "#8a8a8a"},
        },
        yaxis={"showgrid": False, "tickfont": {"color": "#ededed", "size": 13}},
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ---------- Alternative scenario ----------
if implications.alternative:
    st.divider()
    alt = implications.alternative
    alt_color = {
        "Risk-On": "#4a9d6e",
        "Neutral": "#a8a29e",
        "Risk-Off": "#c14040",
    }.get(alt.risk_profile, "#a8a29e")

    st.markdown(
        f"""
        <div style="
            padding: 18px 22px;
            border: 1px solid #232323;
            border-left: 3px solid {alt_color};
            border-radius: 6px;
            background: #0f0f0f;
        ">
            <div style="font-family: 'JetBrains Mono', monospace;
                        font-size: 10.5px; color: #d4a017;
                        letter-spacing: .12em; text-transform: uppercase;
                        margin-bottom: 8px;">
                Alternative scenario  ·  {alt.probability * 100:.0f}% probability
            </div>
            <div style="font-size: 16px; font-weight: 500; color: #ededed;
                        margin-bottom: 6px;">
                {alt.regime_label} ({alt.risk_profile})
            </div>
            <div style="font-size: 14px; color: #a3a3a3; line-height: 1.6;">
                {alt.headline_change}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------- Caveats ----------
st.divider()
with st.expander("Caveats and assumptions", expanded=False):
    for c in implications.caveats:
        st.markdown(f"- {c}")

st.caption(f"Data source: **{implications.data_source}** · Generated for {implications.as_of}")
