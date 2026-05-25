"""Live data path for the API — reads from the warehouse + HMM labels.

Only invoked when both files are present (the same check is in
`regime.implications.service` and `regime.api.main._current_data_source`).
Falls back to `regime.api.sample` otherwise.

Each function returns a plain dict in the same shape as its `sample.*` peer
so the route handlers in main.py can swap modules without restructuring.
"""

from __future__ import annotations

import json
from datetime import date as _date
from pathlib import Path
from typing import Any

from regime.config import get_settings

REGIME_LABELS = {
    0: "Bull / low-vol",
    1: "Neutral / chop",
    2: "Bear / high-vol",
    3: "Tail / crisis",  # if BIC chose K=4 the 4th state shows up
}


def _labels_path() -> Path:
    return get_settings().data_dir / "models" / "hmm" / "labels.parquet"


def _spy_price_at(con: Any, target: _date) -> float | None:
    """Last SPY adj_close on or before `target`."""
    row = con.execute(
        f"""
        select adj_close
        from main_marts.mart_features
        where ticker = 'SPY' and trade_date <= '{target.isoformat()}'
        order by trade_date desc
        limit 1
        """
    ).fetchone()
    return float(row[0]) if row else None


def _build_probs(regime: int) -> dict[str, float]:
    """Same one-hot-with-smoothing proxy used elsewhere when only argmax is stored.

    If the HMM picked K=4 (a "tail/crisis" 4th state), we collapse that state
    into the bear bucket (state 2) so the wire format stays canonically 3-state.
    Probabilities are normalised AFTER the collapse so they always sum to 1.0.
    """
    canonical = min(regime, 2)
    base = {0: 0.05, 1: 0.05, 2: 0.05}
    base[canonical] = 0.85
    if canonical == 0:
        base[1] += 0.05
    elif canonical == 1:
        base[0] += 0.025
        base[2] += 0.025
    else:
        base[1] += 0.05
    total = sum(base.values())
    return {str(k): round(v / total, 4) for k, v in base.items()}


def _label_for(regime: int) -> str:
    # Collapse K=4 tail state into "bear" for the wire format
    canonical = 2 if regime >= 2 else regime
    return REGIME_LABELS[canonical]


def latest_regime() -> dict[str, Any]:
    import duckdb
    import polars as pl

    labels = pl.read_parquet(_labels_path()).sort("feature_date")
    latest = labels.tail(1).to_dicts()[0]
    regime = int(latest["regime_state"])
    target = latest["feature_date"]

    con = duckdb.connect(str(get_settings().duckdb_path), read_only=True)
    try:
        price = _spy_price_at(con, target)
    finally:
        con.close()

    return {
        "date": target.isoformat(),
        "regime": min(regime, 2),  # canonical 3 states externally
        "regime_label": _label_for(regime),
        "probabilities": _build_probs(regime),
        "price": round(price, 2) if price else 0.0,
    }


def regime_for_date(d: _date) -> dict[str, Any] | None:
    import duckdb
    import polars as pl

    labels = pl.read_parquet(_labels_path())
    row = labels.filter(pl.col("feature_date") == d)
    if row.height == 0:
        return None
    regime = int(row["regime_state"][0])

    con = duckdb.connect(str(get_settings().duckdb_path), read_only=True)
    try:
        price = _spy_price_at(con, d)
    finally:
        con.close()

    return {
        "date": d.isoformat(),
        "regime": min(regime, 2),
        "regime_label": _label_for(regime),
        "probabilities": _build_probs(regime),
        "price": round(price, 2) if price else 0.0,
    }


def regime_history(start: _date | None, end: _date | None, limit: int) -> list[dict[str, Any]]:
    import duckdb
    import polars as pl

    labels = pl.read_parquet(_labels_path()).sort("feature_date")
    if start:
        labels = labels.filter(pl.col("feature_date") >= start)
    if end:
        labels = labels.filter(pl.col("feature_date") <= end)
    labels = labels.tail(limit)  # most-recent N

    con = duckdb.connect(str(get_settings().duckdb_path), read_only=True)
    try:
        # .fetchall() returns plain tuples — more portable across the
        # duckdb/polars version matrix than .pl() (which has had interop
        # regressions). Each row is (date, float).
        rows = con.execute(
            """
            select trade_date, adj_close
            from main_marts.mart_features
            where ticker = 'SPY'
            order by trade_date
            """
        ).fetchall()
    finally:
        con.close()

    px_map: dict[_date, float] = {r[0]: float(r[1]) for r in rows}
    out: list[dict[str, Any]] = []
    for r in labels.iter_rows(named=True):
        d = r["feature_date"]
        regime = int(r["regime_state"])
        out.append(
            {
                "date": d.isoformat(),
                "regime": min(regime, 2),
                "regime_label": _label_for(regime),
                "probabilities": _build_probs(regime),
                "price": round(px_map.get(d, 0.0), 2),
            }
        )
    return out


def regime_distribution() -> dict[str, Any]:
    import polars as pl

    labels = pl.read_parquet(_labels_path()).sort("feature_date")
    states_series = labels["regime_state"]
    total = len(states_series)
    counts: dict[int, int] = {0: 0, 1: 0, 2: 0}
    for s_raw in states_series:
        canonical = min(int(s_raw), 2)
        counts[canonical] = counts.get(canonical, 0) + 1

    return {
        "as_of": labels.tail(1).to_dicts()[0]["feature_date"].isoformat(),
        "total_days": total,
        "states": [
            {
                "state": s,
                "label": REGIME_LABELS[s],
                "n_days": counts[s],
                "pct": round(100 * counts[s] / total, 1) if total else 0,
            }
            for s in (0, 1, 2)
        ],
    }


def backtest_summary() -> dict[str, Any]:
    """Read the latest summary_latest.json written by `regime-backtest all`."""
    settings = get_settings()
    path = settings.data_dir / "backtests" / "summary_latest.json"
    if not path.exists():
        # No backtest output yet — caller should fall back to sample.
        raise FileNotFoundError(path)

    raw = json.loads(path.read_text())
    # On-disk shape is a list of BacktestSummary; reshape to API contract
    name_to_label = {
        "baseline_trend": "Trend (unconditional)",
        "regime_conditioned_trend": "Trend + Regime Overlay",
        "regime_conditioned_meanrev": "Mean-Rev + Regime",
    }
    strategies = []
    for s in raw:
        n = s["strategy"]
        strategies.append(
            {
                "name": n,
                "label": name_to_label.get(n, n),
                "sharpe": round(s["sharpe"], 2),
                "sortino": round(s["sortino"], 2),
                "cagr_pct": round(100 * s["cagr"], 2),
                "max_dd_pct": round(100 * s["max_drawdown"], 2),
                "calmar": round(s["calmar"], 2),
                "final_equity": round(
                    s["capital"]
                    * (1 + s["cagr"])
                    ** (
                        (_date.fromisoformat(s["end"]) - _date.fromisoformat(s["start"])).days
                        / 365.25
                    ),
                    2,
                ),
            }
        )

    # Sharpe improvement: regime-conditioned-trend vs baseline-trend
    base_sharpe = next((x["sharpe"] for x in strategies if x["name"] == "baseline_trend"), 1.0)
    cond_sharpe = next(
        (x["sharpe"] for x in strategies if x["name"] == "regime_conditioned_trend"), base_sharpe
    )
    improvement = round(cond_sharpe / base_sharpe, 2) if base_sharpe else 1.0

    return {
        "as_of": raw[0]["end"] if raw else "",
        "strategies": strategies,
        "sharpe_improvement": improvement,
        "note": (
            "Live numbers from the walk-forward backtest over 2018-01-02 to the most "
            "recent ingested date. 1bp linear slippage, configurable borrow drag on "
            "shorts. Source: data/backtests/summary_latest.json."
        ),
    }
