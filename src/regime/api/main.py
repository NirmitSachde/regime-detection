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
from regime.config import get_settings
from regime.implications import (
    RegimeImplications,
    get_implications_for_date,
    get_latest_implications,
)


def _current_data_source() -> str:
    """Return 'warehouse' if a populated DuckDB exists, else 'synthetic'.

    The same module powers all endpoints; this single check decides
    whether responses surface real pipeline output or the baked-in
    sample data. On Render's free-tier container there is no warehouse,
    so this always returns 'synthetic' there.
    """
    settings = get_settings()
    labels_path = settings.data_dir / "models" / "hmm" / "labels.parquet"
    if settings.duckdb_path.exists() and labels_path.exists():
        return "warehouse"
    return "synthetic"


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
# Allow the Pages site explicitly + wildcard so curl + other origins still work.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://nirmitsachde.github.io",
        "*",
    ],
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
    data_source: str = Field(
        description="'warehouse' if served from real ingested data, 'synthetic' otherwise"
    )


class RegimeHistory(BaseModel):
    n: int
    items: list[RegimeClassification]
    data_source: str


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
    data_source: str


class RegimeStateCount(BaseModel):
    state: int
    label: str
    n_days: int
    pct: float


class RegimeDistribution(BaseModel):
    as_of: str
    total_days: int
    states: list[RegimeStateCount]
    data_source: str


class APIIndex(BaseModel):
    name: str
    version: str
    description: str
    docs: dict[str, str]
    endpoints: dict[str, str]


# ---------- Routes ----------


@app.get("/", response_model=APIIndex, tags=["meta"])
def index() -> APIIndex:
    """Friendly landing payload — list of available endpoints + doc links.

    FastAPI returns 404 at `/` by default. This handler exists so that
    anyone curling the base URL gets something useful instead of an error.
    """
    return APIIndex(
        name="regime-detection API",
        version=__version__,
        description=(
            "Read-only API for the adaptive market-regime detection pipeline. "
            "Open /docs for interactive Swagger, /redoc for ReDoc."
        ),
        docs={
            "swagger": "/docs",
            "redoc": "/redoc",
            "openapi": "/openapi.json",
        },
        endpoints={
            "GET /health": "Liveness probe",
            "GET /regime/latest": "Most recent regime classification + probabilities",
            "GET /regime/{date}": "Regime for a specific YYYY-MM-DD",
            "GET /regime/history": "Time-series of classifications (?start=&end=&limit=)",
            "GET /regime/distribution": "How many days spent in each regime",
            "GET /regime/implications/latest": "PM-facing allocation guidance",
            "GET /regime/implications/{date}": "Allocation guidance for a specific date",
            "GET /backtest/summary": "Risk-adjusted stats for all three strategies",
        },
    )


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    """Liveness probe. Returns 200 when the service is up."""
    return HealthResponse(status="ok", version=__version__)


@app.get("/regime/latest", response_model=RegimeClassification, tags=["regime"])
def regime_latest() -> RegimeClassification:
    """Most recent regime classification with probability vector."""
    data = sample.latest_regime()
    return RegimeClassification(**data, data_source=_current_data_source())


# NOTE: static-path routes (/regime/history, /regime/distribution,
# /regime/implications/...) MUST be declared BEFORE the catch-all
# /regime/{day} route, otherwise FastAPI matches "history"/"distribution"
# as the {day} path param and rejects with 422.


@app.get("/regime/history", response_model=RegimeHistory, tags=["regime"])
def regime_history(
    start: Annotated[date | None, Query(description="Inclusive start date")] = None,
    end: Annotated[date | None, Query(description="Inclusive end date")] = None,
    limit: Annotated[int, Query(ge=1, le=2000, description="Max rows")] = 365,
) -> RegimeHistory:
    """Time-series of the most-recent `limit` classifications. Caps at 2000."""
    items = sample.regime_history(start, end, limit)
    src = _current_data_source()
    return RegimeHistory(
        n=len(items),
        items=[RegimeClassification(**r, data_source=src) for r in items],
        data_source=src,
    )


@app.get("/regime/distribution", response_model=RegimeDistribution, tags=["regime"])
def regime_distribution() -> RegimeDistribution:
    """How much time was spent in each regime, in the served data window."""
    data = sample.regime_distribution()
    return RegimeDistribution(
        as_of=data["as_of"],
        total_days=data["total_days"],
        states=[RegimeStateCount(**s) for s in data["states"]],
        data_source=_current_data_source(),
    )


@app.get("/regime/{day}", response_model=RegimeClassification, tags=["regime"])
def regime_for_day(day: date) -> RegimeClassification:
    """Regime classification for a specific date (`YYYY-MM-DD`)."""
    data = sample.regime_for_date(day)
    if data is None:
        raise HTTPException(
            status_code=404,
            detail=f"No regime classification available for {day.isoformat()}",
        )
    return RegimeClassification(**data, data_source=_current_data_source())


@app.get(
    "/regime/implications/latest",
    response_model=RegimeImplications,
    tags=["implications"],
)
def implications_latest() -> RegimeImplications:
    """PM-facing allocation guidance for the most recent regime classification.

    Returns the regime, confidence, historical performance, recommended
    asset-class tilts (in bps), and a plain-English headline suitable for
    a daily PM brief.
    """
    return get_latest_implications()


@app.get(
    "/regime/implications/{day}",
    response_model=RegimeImplications,
    tags=["implications"],
)
def implications_for_day(day: date) -> RegimeImplications:
    """Allocation guidance for a specific date (`YYYY-MM-DD`)."""
    payload = get_implications_for_date(day)
    if payload is None:
        raise HTTPException(
            status_code=404,
            detail=f"No regime classification available for {day.isoformat()}",
        )
    return payload


@app.get("/backtest/summary", response_model=BacktestSummary, tags=["backtest"])
def backtest_summary() -> BacktestSummary:
    """Risk-adjusted summary stats for all three strategies."""
    data = sample.backtest_summary()
    return BacktestSummary(
        as_of=data["as_of"],
        strategies=[StrategyStats(**s) for s in data["strategies"]],
        sharpe_improvement=data["sharpe_improvement"],
        note=data["note"],
        data_source=_current_data_source(),
    )


# ---------- Run with uvicorn ----------
# uv run uvicorn regime.api.main:app --reload
