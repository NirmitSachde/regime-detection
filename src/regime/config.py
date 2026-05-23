"""Centralized, type-safe configuration via pydantic-settings."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Data paths
    data_dir: Path = Field(default=Path("./data"))
    duckdb_path: Path = Field(default=Path("./data/warehouse.duckdb"))
    raw_dir: Path = Field(default=Path("./data/raw"))

    # External APIs
    fred_api_key: str = Field(default="")

    # MLflow
    mlflow_tracking_uri: str = Field(default="file:./data/mlruns")
    mlflow_experiment_name: str = Field(default="regime-detection")

    # Prefect
    prefect_api_url: str = Field(default="http://localhost:4200/api")

    # Drift
    ntfy_topic: str = Field(default="")

    # Logging
    log_level: str = Field(default="INFO")

    # Determinism
    random_seed: int = Field(default=42)

    # Universe + history
    universe: str = Field(
        default="SPY,QQQ,IWM,DIA,XLK,XLF,XLE,XLY,XLP,XLV,XLI,XLU,XLB,XLRE,XLC"
    )
    history_start: str = Field(default="2010-01-01")

    @field_validator("data_dir", "raw_dir", mode="after")
    @classmethod
    def _ensure_dir(cls, v: Path) -> Path:
        v.mkdir(parents=True, exist_ok=True)
        return v

    @property
    def universe_list(self) -> list[str]:
        return [t.strip().upper() for t in self.universe.split(",") if t.strip()]

    @property
    def prices_dir(self) -> Path:
        d = self.raw_dir / "prices"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def macro_dir(self) -> Path:
        d = self.raw_dir / "macro"
        d.mkdir(parents=True, exist_ok=True)
        return d


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
