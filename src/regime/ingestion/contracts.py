"""Pydantic v2 data contracts for raw ingested rows.

Every external row crosses a Pydantic boundary on the way into the warehouse.
This is the single source of truth for raw-zone schemas.
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, Field, field_validator


class OHLCVBar(BaseModel):
    """One day of OHLCV for one ticker (adjusted for splits/dividends)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ticker: str = Field(min_length=1, max_length=10)
    date: date
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    adj_close: float = Field(gt=0)
    volume: float = Field(ge=0)

    @field_validator("ticker", mode="before")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.strip().upper()

    @field_validator("high")
    @classmethod
    def _high_ge_low(cls, v: float, info: object) -> float:
        return v


class MacroSeriesPoint(BaseModel):
    """One observation of a FRED macro series."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    series_id: str = Field(min_length=1, max_length=32)
    date: date
    value: float | None  # FRED can return missing values for unreleased prints

    @field_validator("series_id", mode="before")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.strip().upper()


class IngestionRunStats(BaseModel):
    """Per-flow summary for observability + idempotency proofs."""

    model_config = ConfigDict(extra="forbid")

    flow: str
    started_at: str
    finished_at: str
    rows_in: int
    rows_written: int
    files_written: int
    bytes_written: int
    new_or_changed: bool
