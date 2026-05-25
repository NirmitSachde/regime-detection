"""Historical performance statistics per regime, computed from the warehouse.

The Streamlit page asks "what happened the last N times we were in regime X?"
This module answers that with a small set of distributional stats, computed
from SPY (or configurable proxy) daily returns over the labelled history.
"""

from __future__ import annotations

import math
import statistics
from pathlib import Path
from typing import TYPE_CHECKING

from regime.implications.models import HistoricalRegimeStats
from regime.implications.policy import REGIME_LABELS

if TYPE_CHECKING:
    import polars as pl

TRADING_DAYS_PER_YEAR = 252
_DEFAULT_PROXY = "SPY"


def _episode_runs(states: list[int]) -> list[tuple[int, int]]:
    """List of (state, run_length) tuples preserving order."""
    if not states:
        return []
    out: list[tuple[int, int]] = []
    cur, n = states[0], 1
    for s in states[1:]:
        if s == cur:
            n += 1
        else:
            out.append((cur, n))
            cur, n = s, 1
    out.append((cur, n))
    return out


def _max_drawdown(returns: list[float]) -> float:
    """Worst peak-to-trough drawdown over the series, as a fraction (<=0)."""
    if not returns:
        return 0.0
    equity = 1.0
    peak = 1.0
    worst = 0.0
    for r in returns:
        equity *= 1.0 + r
        if equity > peak:
            peak = equity
        dd = (equity - peak) / peak
        if dd < worst:
            worst = dd
    return worst


def historical_stats_for_regime(
    labels: pl.DataFrame,
    returns: pl.DataFrame,
    regime: int,
    proxy: str = _DEFAULT_PROXY,
) -> HistoricalRegimeStats:
    """Aggregate historical performance for `regime`.

    Args:
        labels:  columns `feature_date`, `regime_state`
        returns: columns `trade_date`, `log_ret_1d` (or `ret_1d`) for a single ticker
        regime:  the target regime label (0/1/2)
        proxy:   the ticker name to record on the response
    """
    import polars as pl  # noqa: PLC0415 - lazy import so slim API container can skip polars

    if labels.height == 0 or returns.height == 0:
        return _empty_stats(proxy)

    # Pick the return column we have. Prefer simple returns; fall back to log.
    ret_col = "log_ret_1d" if "log_ret_1d" in returns.columns else "ret_1d"
    if ret_col not in returns.columns:
        return _empty_stats(proxy)

    joined = (
        labels.rename({"feature_date": "trade_date"})
        .join(returns.select(["trade_date", ret_col]), on="trade_date", how="inner")
        .drop_nulls(subset=[ret_col])
    ).sort("trade_date")

    if joined.height == 0:
        return _empty_stats(proxy)

    in_regime = joined.filter(pl.col("regime_state") == regime)

    if in_regime.height == 0:
        return HistoricalRegimeStats(
            n_episodes=0,
            total_days=0,
            avg_duration_days=0.0,
            median_daily_return_pct=0.0,
            annualized_return_pct=0.0,
            annualized_vol_pct=0.0,
            hit_rate_pct=0.0,
            max_drawdown_pct=0.0,
            sample_basis=f"{proxy} adj close",
        )

    rets = in_regime[ret_col].to_list()
    # If returns are logs, convert to simple for stats people read more easily
    if ret_col == "log_ret_1d":
        simple = [math.exp(r) - 1.0 for r in rets]
    else:
        simple = list(rets)

    median_ret = statistics.median(simple)
    mean_ret = statistics.mean(simple)
    vol = statistics.pstdev(simple) if len(simple) > 1 else 0.0
    ann_ret = (1.0 + mean_ret) ** TRADING_DAYS_PER_YEAR - 1.0 if mean_ret > -1 else -1.0
    ann_vol = vol * math.sqrt(TRADING_DAYS_PER_YEAR)
    hit_rate = sum(1 for r in simple if r > 0) / len(simple)
    mdd = _max_drawdown(simple)

    runs = _episode_runs(joined["regime_state"].to_list())
    episode_lengths = [n for s, n in runs if s == regime]
    n_episodes = len(episode_lengths)
    total_days = sum(episode_lengths)
    avg_dur = statistics.mean(episode_lengths) if episode_lengths else 0.0

    return HistoricalRegimeStats(
        n_episodes=n_episodes,
        total_days=total_days,
        avg_duration_days=round(avg_dur, 1),
        median_daily_return_pct=round(100 * median_ret, 3),
        annualized_return_pct=round(100 * ann_ret, 2),
        annualized_vol_pct=round(100 * ann_vol, 2),
        hit_rate_pct=round(100 * hit_rate, 1),
        max_drawdown_pct=round(100 * mdd, 2),
        sample_basis=f"{proxy} adj close",
    )


def days_in_current_run(labels: "pl.DataFrame") -> int | None:
    """How many consecutive most-recent days share the latest regime."""
    if labels.height == 0:
        return None
    sorted_lbls = labels.sort("feature_date")
    states = sorted_lbls["regime_state"].to_list()
    if not states:
        return None
    latest = states[-1]
    n = 0
    for s in reversed(states):
        if s == latest:
            n += 1
        else:
            break
    return n


def load_labels_and_returns(
    labels_path: Path,
    duckdb_path: Path,
    proxy: str = _DEFAULT_PROXY,
) -> "tuple[pl.DataFrame, pl.DataFrame] | None":
    """Read HMM labels + per-ticker returns from disk. Returns None if missing."""
    if not labels_path.exists() or not duckdb_path.exists():
        return None

    import duckdb  # noqa: PLC0415 - lazy
    import polars as pl  # noqa: PLC0415 - lazy

    labels = pl.read_parquet(labels_path)

    con = duckdb.connect(str(duckdb_path), read_only=True)
    try:
        returns = con.execute(
            f"""
            select trade_date, log_ret_1d
            from main_intermediate.int_returns
            where ticker = '{proxy}'
              and log_ret_1d is not null
            """
        ).pl()
    except Exception:
        return None
    finally:
        con.close()

    return labels, returns


def _empty_stats(proxy: str) -> HistoricalRegimeStats:
    return HistoricalRegimeStats(
        n_episodes=0,
        total_days=0,
        avg_duration_days=0.0,
        median_daily_return_pct=0.0,
        annualized_return_pct=0.0,
        annualized_vol_pct=0.0,
        hit_rate_pct=0.0,
        max_drawdown_pct=0.0,
        sample_basis=f"{proxy} adj close",
    )


# Synthetic stats used when the warehouse is empty. Values reflect what we'd
# expect to see in each regime; the Streamlit page tags these as 'synthetic'.
SYNTHETIC_STATS: dict[int, HistoricalRegimeStats] = {
    0: HistoricalRegimeStats(
        n_episodes=4,
        total_days=1185,
        avg_duration_days=296.2,
        median_daily_return_pct=0.078,
        annualized_return_pct=21.4,
        annualized_vol_pct=11.9,
        hit_rate_pct=56.8,
        max_drawdown_pct=-9.4,
        sample_basis="SPY adj close (synthetic)",
    ),
    1: HistoricalRegimeStats(
        n_episodes=3,
        total_days=369,
        avg_duration_days=123.0,
        median_daily_return_pct=0.012,
        annualized_return_pct=3.6,
        annualized_vol_pct=19.8,
        hit_rate_pct=51.2,
        max_drawdown_pct=-14.7,
        sample_basis="SPY adj close (synthetic)",
    ),
    2: HistoricalRegimeStats(
        n_episodes=3,
        total_days=296,
        avg_duration_days=98.7,
        median_daily_return_pct=-0.085,
        annualized_return_pct=-19.2,
        annualized_vol_pct=41.2,
        hit_rate_pct=44.3,
        max_drawdown_pct=-34.1,
        sample_basis="SPY adj close (synthetic)",
    ),
}


def synthetic_stats(regime: int) -> HistoricalRegimeStats:
    return SYNTHETIC_STATS.get(regime, _empty_stats("SPY"))


def regime_label(regime: int) -> str:
    return REGIME_LABELS.get(regime, "Unknown")
