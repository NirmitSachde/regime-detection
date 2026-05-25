"""Weekly drift check — score last week's predictions and ping ntfy if F1 drops."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import polars as pl
import requests
from prefect import flow, task
from sklearn.metrics import f1_score

from regime.config import get_settings
from regime.logging import get_logger

log = get_logger(__name__)

F1_DROP_THRESHOLD = 0.10  # 10 percentage points


@task
def load_baseline_f1() -> float:
    s = get_settings()
    meta = s.data_dir / "models" / "lgbm" / "metadata.json"
    if not meta.exists():
        log.warning("drift.baseline.missing")
        return 0.0
    return float(json.loads(meta.read_text()).get("overall_macro_f1", 0.0))


@task
def score_last_week() -> float:
    """Placeholder: in production this rescore would use the live booster +
    realized labels (delayed by the HMM's smoothing horizon). For now we
    simulate by reading the labels file and returning the rolling macro-F1
    on the most recent 7 days — wired through to keep the flow end-to-end
    runnable before live scoring infra is built."""
    s = get_settings()
    labels_path = s.data_dir / "models" / "hmm" / "labels.parquet"
    if not labels_path.exists():
        return 0.0

    labels = pl.read_parquet(labels_path).sort("feature_date")
    if labels.height < 14:
        return 0.0

    recent = labels.tail(7)["regime_state"].to_numpy()
    prior = labels.slice(-14, 7)["regime_state"].to_numpy()
    if recent.size == 0 or prior.size == 0:
        return 0.0
    return float(f1_score(prior, recent, average="macro", zero_division=0))


@task
def notify_ntfy(message: str, topic: str | None = None) -> bool:
    s = get_settings()
    topic = topic or s.ntfy_topic
    if not topic:
        log.warning("drift.ntfy.no_topic")
        return False
    url = f"https://ntfy.sh/{topic}"
    try:
        requests.post(url, data=message.encode("utf-8"), timeout=10)
        log.info("drift.ntfy.sent", topic=topic)
        return True
    except requests.RequestException as exc:
        log.error("drift.ntfy.failed", error=str(exc))
        return False


@flow(name="drift-check")
def drift_check_flow() -> dict[str, float | bool | str]:
    baseline = load_baseline_f1()
    current = score_last_week()
    drop = baseline - current
    alert_fired = False
    if drop > F1_DROP_THRESHOLD and baseline > 0:
        ts = datetime.now(UTC).isoformat()
        alert_fired = notify_ntfy(
            f"[regime-detection] drift alert at {ts}\n"
            f"baseline F1={baseline:.3f}, current F1={current:.3f}, drop={drop:.3f}"
        )
    log.info("drift.check.done", baseline=baseline, current=current, drop=drop, alert=alert_fired)
    return {"baseline": baseline, "current": current, "drop": drop, "alert_fired": alert_fired}


if __name__ == "__main__":
    drift_check_flow()
