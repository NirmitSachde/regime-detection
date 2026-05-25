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
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from regime import __version__
from regime.api import live, sample
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


def _data_module() -> object:
    """Pick the live module when warehouse is present, else the sample fallback.

    Both modules expose the same function names + return shapes — the route
    handlers below stay agnostic to the source.
    """
    return live if _current_data_source() == "warehouse" else sample


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


# Catch-all handler so 500s don't disappear into the Render log void.
# Logs the traceback to stderr (visible in Render's log stream) and
# returns a structured JSON error to the client.
@app.exception_handler(Exception)
async def _unhandled_exception_handler(_request: object, exc: Exception) -> JSONResponse:
    import sys
    import traceback

    tb = traceback.format_exc()
    print(f"[api] unhandled exception:\n{tb}", file=sys.stderr, flush=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "error_type": type(exc).__name__,
            "error_message": str(exc)[:200],
        },
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


class NotableEvent(BaseModel):
    date: str
    event: str
    classification: RegimeClassification | None
    note: str | None = None


class NotableEventsResponse(BaseModel):
    description: str
    events: list[NotableEvent]
    data_source: str


# Canonical "interesting moments" for the demo. Chosen to span:
# - Both regimes we'd expect the model to identify clearly (COVID, 2022 bear)
# - Inflection points (2018 Q4, 2020 March, 2022 H1)
# - Pure bull-trend periods (mid-2019, 2024) for contrast
# - Recent stress (SVB 2023, yen unwind 2024)
NOTABLE_DATES: list[tuple[str, str]] = [
    ("2018-10-03", "Q4 2018 selloff begins (Fed hike + China trade)"),
    ("2018-12-24", "Christmas Eve crash trough"),
    ("2019-07-26", "Mid-2019 bull peak (pre yield-curve inversion)"),
    ("2020-02-19", "Pre-COVID S&P 500 peak ($339 SPY)"),
    ("2020-03-23", "COVID crash trough ($224 SPY, -34% in 5 weeks)"),
    ("2020-08-19", "Post-COVID recovery peak"),
    ("2021-11-08", "Late-2021 bull peak before correction"),
    ("2022-01-03", "2022 bear market begins"),
    ("2022-06-13", "Mid-2022 trough (Fed front-loading)"),
    ("2022-10-12", "October 2022 CPI surprise / yields spike"),
    ("2023-03-13", "SVB collapse week"),
    ("2023-10-19", "Treasury yield spike (10Y >5%)"),
    ("2024-08-05", "Yen carry unwind ('Bloody Monday')"),
    ("2024-12-18", "Fed hawkish pivot"),
]


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
            "GET /regime/notable": "Classifications on historically notable trading days",
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
    data = _data_module().latest_regime()  # type: ignore[attr-defined]
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
    items = _data_module().regime_history(start, end, limit)  # type: ignore[attr-defined]
    src = _current_data_source()
    return RegimeHistory(
        n=len(items),
        items=[RegimeClassification(**r, data_source=src) for r in items],
        data_source=src,
    )


@app.get("/regime/distribution", response_model=RegimeDistribution, tags=["regime"])
def regime_distribution() -> RegimeDistribution:
    """How much time was spent in each regime, in the served data window."""
    data = _data_module().regime_distribution()  # type: ignore[attr-defined]
    return RegimeDistribution(
        as_of=data["as_of"],
        total_days=data["total_days"],
        states=[RegimeStateCount(**s) for s in data["states"]],
        data_source=_current_data_source(),
    )


@app.get("/regime/notable", response_model=NotableEventsResponse, tags=["regime"])
def regime_notable() -> NotableEventsResponse:
    """Regime classifications on historically notable trading days.

    Useful as a "smoke test" of the model — does it identify COVID as
    bear / high-vol, 2024 as bull / low-vol, etc.? Each event includes
    the date, a one-line description, and the regime payload for that
    date (or null if outside the training window).
    """
    src = _current_data_source()
    mod = _data_module()
    events: list[NotableEvent] = []
    for iso, description in NOTABLE_DATES:
        try:
            d = date.fromisoformat(iso)
            data = mod.regime_for_date(d)  # type: ignore[attr-defined]
            if data is None:
                events.append(
                    NotableEvent(
                        date=iso,
                        event=description,
                        classification=None,
                        note="No classification available for this date",
                    )
                )
            else:
                events.append(
                    NotableEvent(
                        date=iso,
                        event=description,
                        classification=RegimeClassification(**data, data_source=src),
                    )
                )
        except Exception as exc:
            events.append(
                NotableEvent(
                    date=iso,
                    event=description,
                    classification=None,
                    note=f"error: {type(exc).__name__}",
                )
            )
    return NotableEventsResponse(
        description=(
            "Regime classification on notable trading days. "
            "Sanity-check the model: COVID should land in bear, 2024 in bull, etc."
        ),
        events=events,
        data_source=src,
    )


@app.get("/regime/{day}", response_model=RegimeClassification, tags=["regime"])
def regime_for_day(day: date) -> RegimeClassification:
    """Regime classification for a specific date (`YYYY-MM-DD`)."""
    data = _data_module().regime_for_date(day)  # type: ignore[attr-defined]
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
    src = _current_data_source()
    if src == "warehouse":
        try:
            data = live.backtest_summary()
        except FileNotFoundError:
            # warehouse exists but no backtest output yet — surface synthetic
            data = sample.backtest_summary()
            src = "synthetic"
    else:
        data = sample.backtest_summary()
    return BacktestSummary(
        as_of=data["as_of"],
        strategies=[StrategyStats(**s) for s in data["strategies"]],
        sharpe_improvement=data["sharpe_improvement"],
        note=data["note"],
        data_source=src,
    )


# ---------- Run with uvicorn ----------
# uv run uvicorn regime.api.main:app --reload
