"""FastAPI surface for the regime-detection pipeline.

Endpoints:
    GET /health
    GET /regime/latest
    GET /regime/{date}
    GET /regime/history?start=YYYY-MM-DD&end=YYYY-MM-DD&limit=N
    GET /regime/distribution
    GET /backtest/summary

Interactive docs live at /docs (Swagger) and /redoc.

The API is read-only — there is no auth surface and no mutation endpoint.
If you fork and run live, put it behind a CDN with caching and rate limits.
"""

from __future__ import annotations

from datetime import date
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from regime import __version__
from regime.api import sample

app = FastAPI(
    title="regime-detection API",
    version=__version__,
    description=(
        "Read-only API for the adaptive market-regime detection pipeline. "
        "Returns regime classifications, probability vectors, and backtest summaries. "
        "Source code: https://github.com/<your-handle>/regime-detection"
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# CORS — the static landing site is on a different origin (GH Pages).
# Lock this down to your actual Pages URL once you have one.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "OPTIONS"],
    allow_headers=["*"],
    max_age=3600,
)


# ---------- Response models ----------


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    docs_url: str = "/docs"


class RegimeProbabilities(BaseModel):
    s0: float = Field(alias="0", description="P(bull / low-vol)")
    s1: float = Field(alias="1", description="P(neutral / chop)")
    s2: float = Field(alias="2", description="P(bear / high-vol)")

    model_config = {"populate_by_name": True}


class RegimeClassification(BaseModel):
    date: str
    regime: int = Field(ge=0, le=2)
    regime_label: str
    probabilities: dict[str, float]
    price: float


class RegimeHistory(BaseModel):
    n: int
    items: list[RegimeClassification]


class StrategyStats(BaseModel):
    name: str
    label: str
    sharpe: float
    sortino: float
    cagr_pct: float
    max_dd_pct: float
    calmar: float
    final_equity: float


class BacktestSummary(BaseModel):
    as_of: str
    strategies: list[StrategyStats]
    sharpe_improvement: float
    note: str


class RegimeStateCount(BaseModel):
    state: int
    label: str
    n_days: int
    pct: float


class RegimeDistribution(BaseModel):
    as_of: str
    total_days: int
    states: list[RegimeStateCount]


# ---------- Routes ----------


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    """Liveness probe. Returns 200 when the service is up."""
    return HealthResponse(status="ok", version=__version__)


@app.get("/regime/latest", response_model=RegimeClassification, tags=["regime"])
def regime_latest() -> RegimeClassification:
    """Most recent regime classification with probability vector."""
    data = sample.latest_regime()
    return RegimeClassification(**data)  # type: ignore[arg-type]


@app.get("/regime/{day}", response_model=RegimeClassification, tags=["regime"])
def regime_for_day(day: date) -> RegimeClassification:
    """Regime classification for a specific date (`YYYY-MM-DD`)."""
    data = sample.regime_for_date(day)
    if data is None:
        raise HTTPException(
            status_code=404,
            detail=f"No regime classification available for {day.isoformat()}",
        )
    return RegimeClassification(**data)  # type: ignore[arg-type]


@app.get("/regime/history", response_model=RegimeHistory, tags=["regime"])
def regime_history(
    start: Annotated[date | None, Query(description="Inclusive start date")] = None,
    end: Annotated[date | None, Query(description="Inclusive end date")] = None,
    limit: Annotated[int, Query(ge=1, le=2000, description="Max rows")] = 365,
) -> RegimeHistory:
    """Time-series of regime classifications. Caps at 2000 rows."""
    items = sample.regime_history(start, end, limit)
    return RegimeHistory(
        n=len(items),
        items=[RegimeClassification(**r) for r in items],  # type: ignore[arg-type]
    )


@app.get("/regime/distribution", response_model=RegimeDistribution, tags=["regime"])
def regime_distribution() -> RegimeDistribution:
    """How much time was spent in each regime, in the served data window."""
    data = sample.regime_distribution()
    return RegimeDistribution(
        as_of=data["as_of"],
        total_days=data["total_days"],
        states=[RegimeStateCount(**s) for s in data["states"]],  # type: ignore[arg-type]
    )


@app.get("/backtest/summary", response_model=BacktestSummary, tags=["backtest"])
def backtest_summary() -> BacktestSummary:
    """Risk-adjusted summary stats for all three strategies."""
    data = sample.backtest_summary()
    return BacktestSummary(
        as_of=data["as_of"],
        strategies=[StrategyStats(**s) for s in data["strategies"]],
        sharpe_improvement=data["sharpe_improvement"],
        note=data["note"],
    )


# ---------- Run with uvicorn ----------
# uv run uvicorn regime.api.main:app --reload
