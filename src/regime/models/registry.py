"""Thin MLflow helpers — keep tracking calls in one place."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import mlflow

from regime.config import get_settings
from regime.logging import get_logger

log = get_logger(__name__)


def _configure_mlflow() -> None:
    s = get_settings()
    mlflow.set_tracking_uri(s.mlflow_tracking_uri)
    mlflow.set_experiment(s.mlflow_experiment_name)


@contextmanager
def start_run(name: str, tags: dict[str, str] | None = None) -> Any:
    _configure_mlflow()
    with mlflow.start_run(run_name=name, tags=tags or {}) as run:
        yield run


def log_dict(metrics: dict[str, float] | None = None, params: dict[str, Any] | None = None) -> None:
    if params:
        mlflow.log_params({k: str(v) for k, v in params.items()})
    if metrics:
        for k, v in metrics.items():
            mlflow.log_metric(k, float(v))


def register_model(local_dir: str, name: str, stage: str = "Staging") -> str | None:
    """Register a model artifact and (best-effort) promote to a stage."""
    _configure_mlflow()
    try:
        result = mlflow.register_model(local_dir, name)
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=name, version=result.version, stage=stage, archive_existing_versions=False
        )
        log.info("mlflow.register", name=name, version=result.version, stage=stage)
        return str(result.version)
    except Exception as exc:  # registry is optional in local file-store mode
        log.warning("mlflow.register.failed", name=name, error=str(exc))
        return None
