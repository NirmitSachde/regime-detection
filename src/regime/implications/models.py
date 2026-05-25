"""Pydantic response shapes for the implications module."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, Field


class AllocationTilt(BaseModel):
    """One asset-class recommendation."""

    model_config = ConfigDict(extra="forbid")

    asset_class: str = Field(description="e.g. 'Equity', 'Duration', 'Credit', 'Cash', 'Vol hedge'")
    tilt: str = Field(description="'Overweight' | 'Underweight' | 'Neutral'")
    magnitude: str = Field(description="'Light' | 'Moderate' | 'Strong'")
    bps: int = Field(description="Suggested deviation from benchmark in basis points (signed)")
    rationale: str


class HistoricalRegimeStats(BaseModel):
    """What past instances of this regime looked like."""

    model_config = ConfigDict(extra="forbid")

    n_episodes: int = Field(ge=0)
    total_days: int = Field(ge=0)
    avg_duration_days: float = Field(ge=0)
    median_daily_return_pct: float
    annualized_return_pct: float
    annualized_vol_pct: float = Field(ge=0)
    hit_rate_pct: float = Field(ge=0, le=100)
    max_drawdown_pct: float = Field(le=0)
    sample_basis: str = Field(
        description="What ticker / index the historical stats are computed on (e.g. 'SPY adj close')"
    )


class AlternativeScenario(BaseModel):
    """If confidence is low, what would change under the next-most-likely regime."""

    model_config = ConfigDict(extra="forbid")

    regime: int
    regime_label: str
    probability: float = Field(ge=0, le=1)
    risk_profile: str
    headline_change: str = Field(description="One-line summary of what flips")


class RegimeImplications(BaseModel):
    """Complete PM-facing payload for a single date."""

    model_config = ConfigDict(extra="forbid")

    as_of: date
    regime: int
    regime_label: str
    risk_profile: str = Field(description="'Risk-On' | 'Neutral' | 'Risk-Off'")
    description: str

    confidence: float = Field(ge=0, le=1, description="Top probability from the model")
    confidence_label: str = Field(description="'High' | 'Medium' | 'Low'")
    probabilities: dict[int, float]

    days_in_current_run: int | None = Field(
        default=None,
        description="How many consecutive days the model has been in this regime",
    )

    historical: HistoricalRegimeStats

    allocation: list[AllocationTilt]
    headline: str = Field(description="One-sentence summary suitable for a PM brief")

    alternative: AlternativeScenario | None = Field(
        default=None,
        description="Populated when confidence in the primary regime is below the high-conviction threshold",
    )

    caveats: list[str] = Field(default_factory=list)

    data_source: str = Field(
        description="'live' if read from the warehouse, 'synthetic' if served from sample data"
    )
