"""Baked-in sample data so the API works out of the box without running
the pipeline first. Replace with live warehouse reads once data is in place.

When the warehouse contains real data, `loaders.py` will pull from there
instead of these constants — toggled by the `DATA_DIR` env var pointing
at a populated `data/warehouse.duckdb`.
"""

from __future__ import annotations

import math
from datetime import date, timedelta
from typing import Any

# Deterministic synthetic regime walk — same seed/params as web/sample_data.js
_RNG_SEED = 7
_N = 1850
_START = date(2018, 1, 2)


def _trading_dates(n: int, start: date) -> list[date]:
    out: list[date] = []
    d = start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def _regime_at(i: int, n: int) -> int:
    pct = i / n
    if pct < 0.18:
        return 0
    if pct < 0.22:
        return 2
    if pct < 0.34:
        return 0
    if pct < 0.36:
        return 2
    if pct < 0.55:
        return 0
    if pct < 0.58:
        return 1
    if pct < 0.68:
        return 2
    if pct < 0.78:
        return 1
    if pct < 0.93:
        return 0
    return 1


def _walk_prices() -> tuple[list[date], list[int], list[float]]:
    import random

    random.seed(_RNG_SEED)
    dates = _trading_dates(_N, _START)
    regimes = [_regime_at(i, _N) for i in range(_N)]
    params = {0: (0.00080, 0.0075), 1: (0.00015, 0.0125), 2: (-0.00050, 0.0260)}
    price = 280.0
    prices: list[float] = []
    for r in regimes:
        drift, vol = params[r]
        price *= math.exp(drift + vol * random.gauss(0, 1))
        prices.append(price)
    return dates, regimes, prices


_DATES, _REGIMES, _PRICES = _walk_prices()
_BY_DATE = {d: i for i, d in enumerate(_DATES)}


REGIME_LABELS = {
    0: "Bull / low-vol",
    1: "Neutral / chop",
    2: "Bear / high-vol",
}


def regime_for_date(d: date) -> dict[str, Any] | None:
    """Return regime classification + state probabilities for a date.

    Probabilities are simulated as a soft-max around the chosen state with
    small noise — sufficient to drive a UI, replaced by HMM `predict_proba`
    output once the warehouse is populated.
    """
    i = _BY_DATE.get(d)
    if i is None:
        return None

    state = _REGIMES[i]
    # Build a believable probability vector
    base = [0.05, 0.05, 0.05]
    base[state] = 0.85
    # Lightly diffuse around the chosen state
    if state == 0:
        base[1] += 0.05
    elif state == 1:
        base[0] += 0.025
        base[2] += 0.025
    else:
        base[1] += 0.05
    total = sum(base)
    probs = [round(p / total, 4) for p in base]

    return {
        "date": d.isoformat(),
        "regime": state,
        "regime_label": REGIME_LABELS[state],
        "probabilities": {
            "0": probs[0],
            "1": probs[1],
            "2": probs[2],
        },
        "price": round(_PRICES[i], 2),
    }


def latest_regime() -> dict[str, Any]:
    return regime_for_date(_DATES[-1])  # type: ignore[return-value]


def regime_history(start: date | None, end: date | None, limit: int) -> list[dict[str, Any]]:
    """Return the most-recent `limit` classifications within [start, end].

    Iterates newest-first so a bare `?limit=90` returns the last 90 days,
    not the first 90. The returned list is then re-sorted oldest-first
    so the timeline renders left-to-right in chronological order.
    """
    out: list[dict[str, Any]] = []
    for d in reversed(_DATES):
        if start and d < start:
            continue
        if end and d > end:
            continue
        r = regime_for_date(d)
        if r:
            out.append(r)
        if len(out) >= limit:
            break
    out.reverse()  # chronological for downstream timeline rendering
    return out


def backtest_summary() -> dict[str, Any]:
    """Same numbers shown on the landing page."""
    return {
        "as_of": _DATES[-1].isoformat(),
        "strategies": [
            {
                "name": "buy_hold",
                "label": "Buy & Hold",
                "sharpe": 1.07,
                "sortino": 1.47,
                "cagr_pct": 22.7,
                "max_dd_pct": -37.3,
                "calmar": 0.61,
                "final_equity": 448751.0,
            },
            {
                "name": "baseline_trend",
                "label": "Trend (unconditional)",
                "sharpe": 0.27,
                "sortino": 0.30,
                "cagr_pct": 3.3,
                "max_dd_pct": -44.0,
                "calmar": 0.07,
                "final_equity": 126531.0,
            },
            {
                "name": "regime_conditioned",
                "label": "Trend + Regime Overlay",
                "sharpe": 0.62,
                "sortino": 0.80,
                "cagr_pct": 7.5,
                "max_dd_pct": -28.0,
                "calmar": 0.27,
                "final_equity": 169912.0,
            },
        ],
        "sharpe_improvement": round(0.62 / 0.27, 2),
        "note": "Synthetic-data demo numbers. Live numbers replace these once the pipeline runs against real ingested data.",
    }


def regime_distribution() -> dict[str, Any]:
    counts = {0: 0, 1: 0, 2: 0}
    for r in _REGIMES:
        counts[r] += 1
    total = sum(counts.values())
    return {
        "as_of": _DATES[-1].isoformat(),
        "total_days": total,
        "states": [
            {
                "state": s,
                "label": REGIME_LABELS[s],
                "n_days": counts[s],
                "pct": round(100 * counts[s] / total, 1),
            }
            for s in (0, 1, 2)
        ],
    }
