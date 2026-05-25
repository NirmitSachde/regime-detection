"""FastAPI surface smoke tests against sample data."""

from __future__ import annotations

from fastapi.testclient import TestClient

from regime.api.main import app

client = TestClient(app)


def test_index_lists_endpoints() -> None:
    r = client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert body["name"] == "regime-detection API"
    assert "/regime/latest" in " ".join(body["endpoints"].keys())
    assert body["docs"]["swagger"] == "/docs"


def test_health() -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "version" in body


def test_regime_latest_shape() -> None:
    r = client.get("/regime/latest")
    assert r.status_code == 200
    body = r.json()
    assert "date" in body
    assert body["regime"] in {0, 1, 2}
    assert set(body["probabilities"].keys()) == {"0", "1", "2"}
    total = sum(body["probabilities"].values())
    assert 0.99 < total < 1.01  # rounded probabilities sum to ~1


def test_regime_history_limit_enforced() -> None:
    r = client.get("/regime/history?limit=10")
    assert r.status_code == 200
    body = r.json()
    assert body["n"] <= 10
    assert len(body["items"]) == body["n"]


def test_regime_unknown_date_returns_404() -> None:
    r = client.get("/regime/1900-01-01")
    assert r.status_code == 404


def test_regime_invalid_date_format_returns_422() -> None:
    r = client.get("/regime/not-a-date")
    assert r.status_code == 422


def test_regime_distribution_sums() -> None:
    r = client.get("/regime/distribution")
    assert r.status_code == 200
    body = r.json()
    total_pct = sum(s["pct"] for s in body["states"])
    assert 99.5 < total_pct < 100.5  # rounding leaves a tiny residual
    assert sum(s["n_days"] for s in body["states"]) == body["total_days"]


def test_backtest_summary_has_three_strategies() -> None:
    r = client.get("/backtest/summary")
    assert r.status_code == 200
    body = r.json()
    names = {s["name"] for s in body["strategies"]}
    assert names == {"buy_hold", "baseline_trend", "regime_conditioned"}
    assert body["sharpe_improvement"] > 1.0


def test_cors_allowed() -> None:
    # FastAPI's CORS middleware responds to actual GETs with the headers;
    # OPTIONS preflight is also handled but easier to assert on a real request.
    r = client.get("/health", headers={"Origin": "https://example.com"})
    assert r.status_code == 200
    assert r.headers.get("access-control-allow-origin") == "*"
