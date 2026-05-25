"""Hand-coded allocation policy per regime.

This is the *opinionated* part of the system. Everything upstream — features,
HMM, classifier — is mechanical. This file is where we translate a state label
into a recommendation. Two principles:

1. **Defensibility over cleverness.** The tilts here are textbook macro
   responses to the regime characteristics. A CIO should nod at them and
   move on, not stop to argue about magnitudes.

2. **Magnitudes are stated, not implied.** Every tilt has a bps deviation
   from benchmark. "Overweight equity" with no number is useless to a PM.

The 60/40 benchmark assumed throughout: 60% equity, 30% duration (US Treasury
intermediate), 10% credit (IG corporates). Cash and vol-hedge come out of the
equity bucket when invoked.
"""

from __future__ import annotations

from regime.implications.models import AllocationTilt

REGIME_LABELS: dict[int, str] = {
    0: "Bull / low-vol",
    1: "Neutral / chop",
    2: "Bear / high-vol",
}

REGIME_DESCRIPTIONS: dict[int, str] = {
    0: (
        "Low realised volatility, positive trend, compressing credit spreads. "
        "Macro is supportive: curve term-premia stable, dollar range-bound."
    ),
    1: (
        "Mixed signals. Volatility elevated relative to bull but not extreme; "
        "trend is unclear. Macro readings disagree across indicators."
    ),
    2: (
        "High realised and implied volatility, negative momentum, widening credit "
        "spreads. Flight-to-quality flows into Treasuries. Dollar typically bid."
    ),
}

RISK_PROFILE: dict[int, str] = {
    0: "Risk-On",
    1: "Neutral",
    2: "Risk-Off",
}

# bps deviation from a 60/30/10 (equity/duration/credit) benchmark, with cash and
# vol-hedge taken from the equity sleeve when activated.
ALLOCATION_BY_REGIME: dict[int, list[AllocationTilt]] = {
    0: [
        AllocationTilt(
            asset_class="Equity",
            tilt="Overweight",
            magnitude="Moderate",
            bps=+800,
            rationale=(
                "Trend is your friend; realised vol below long-run average historically "
                "associates with positive forward 1-3 month equity returns."
            ),
        ),
        AllocationTilt(
            asset_class="Duration",
            tilt="Underweight",
            magnitude="Light",
            bps=-300,
            rationale=(
                "Risk-on episodes tend to coincide with mild rate drift higher; "
                "duration provides less diversification when equity-bond correlation "
                "turns positive."
            ),
        ),
        AllocationTilt(
            asset_class="Credit",
            tilt="Overweight",
            magnitude="Moderate",
            bps=+400,
            rationale=(
                "HY OAS compression is the textbook bull-regime carry trade; "
                "spread duration earns its keep here."
            ),
        ),
        AllocationTilt(
            asset_class="Cash",
            tilt="Underweight",
            magnitude="Light",
            bps=-500,
            rationale="Holding cash in a low-vol bull is opportunity cost.",
        ),
        AllocationTilt(
            asset_class="Vol hedge",
            tilt="Underweight",
            magnitude="Light",
            bps=-400,
            rationale=(
                "Long-vol carry is expensive when realised vol stays low; "
                "harvest only minimum tail protection."
            ),
        ),
    ],
    1: [
        AllocationTilt(
            asset_class="Equity",
            tilt="Neutral",
            magnitude="Light",
            bps=0,
            rationale=(
                "No edge in either direction; preserve dry powder for the next regime change."
            ),
        ),
        AllocationTilt(
            asset_class="Duration",
            tilt="Neutral",
            magnitude="Light",
            bps=+100,
            rationale=(
                "Slight tilt to duration for diversification, but no view; "
                "watch the curve for the next signal."
            ),
        ),
        AllocationTilt(
            asset_class="Credit",
            tilt="Neutral",
            magnitude="Light",
            bps=0,
            rationale="Spread carry roughly fair; avoid concentration.",
        ),
        AllocationTilt(
            asset_class="Cash",
            tilt="Overweight",
            magnitude="Light",
            bps=+300,
            rationale=(
                "Hold a buffer to deploy decisively when the regime resolves; "
                "money-market yields make this less costly than usual."
            ),
        ),
        AllocationTilt(
            asset_class="Vol hedge",
            tilt="Neutral",
            magnitude="Light",
            bps=+200,
            rationale=(
                "Add a modest long-vol position — convexity is most valuable "
                "right before a regime shift, not during one."
            ),
        ),
    ],
    2: [
        AllocationTilt(
            asset_class="Equity",
            tilt="Underweight",
            magnitude="Strong",
            bps=-1200,
            rationale=(
                "High-vol bear regimes historically deliver negative median equity "
                "returns; the asymmetry of large drawdowns argues for reducing "
                "exposure decisively, not partially."
            ),
        ),
        AllocationTilt(
            asset_class="Duration",
            tilt="Overweight",
            magnitude="Strong",
            bps=+700,
            rationale=(
                "Flight-to-quality bid for US Treasuries; equity-bond correlation "
                "typically reverts to negative in risk-off, restoring duration's "
                "diversification value."
            ),
        ),
        AllocationTilt(
            asset_class="Credit",
            tilt="Underweight",
            magnitude="Strong",
            bps=-600,
            rationale=(
                "Spreads widen and default risk repricing accelerates; reduce HY "
                "and lower-quality IG exposure first."
            ),
        ),
        AllocationTilt(
            asset_class="Cash",
            tilt="Overweight",
            magnitude="Moderate",
            bps=+600,
            rationale=(
                "Optionality to deploy into dislocations is worth more in a "
                "high-vol regime than at any other time."
            ),
        ),
        AllocationTilt(
            asset_class="Vol hedge",
            tilt="Overweight",
            magnitude="Moderate",
            bps=+500,
            rationale=(
                "VIX term structure may be in backwardation; out-of-the-money put "
                "spreads or VIX call calendars to hedge the left tail."
            ),
        ),
    ],
}


def headline_for_regime(regime: int, confidence: float) -> str:
    """One-sentence summary suitable for a PM brief."""
    rl = REGIME_LABELS.get(regime, "Unknown")
    rp = RISK_PROFILE.get(regime, "Unknown")
    conf_pct = round(confidence * 100)
    return (
        f"Current regime: {rp} ({rl.lower()}), {conf_pct}% model confidence. "
        f"Suggested tilt: {_one_line_tilt(regime)}."
    )


def _one_line_tilt(regime: int) -> str:
    if regime == 0:
        return "overweight equity and credit, underweight duration and cash"
    if regime == 1:
        return "stay close to benchmark, hold a cash buffer, add a modest vol hedge"
    return "reduce equity and credit, increase duration and cash, add vol protection"


def confidence_label(p: float) -> str:
    if p >= 0.75:
        return "High"
    if p >= 0.50:
        return "Medium"
    return "Low"


def caveats_for(regime: int, confidence: float) -> list[str]:
    out = [
        "Regime classifications are a model output, not a forecast. They describe what "
        "regime the data currently *resembles*, not what regime will hold next week.",
        "Historical stats summarise past instances of this regime in the training window; "
        "future occurrences may differ.",
        "Allocation tilts are illustrative and assume a 60/30/10 (equity/duration/credit) "
        "benchmark with cash and vol-hedge sleeves available.",
    ]
    if confidence_label(confidence) != "High":
        out.append(
            "Model confidence is not high. Treat the alternative-scenario card as a "
            "second-most-likely path and consider sizing the tilt down accordingly."
        )
    if regime == 2:
        out.append(
            "Risk-off allocations have higher implementation costs (wider spreads, "
            "thinner liquidity). Rebalance in tranches, not all at once."
        )
    return out
