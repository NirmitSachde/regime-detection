"""Shared pytest fixtures.

Forces all unit tests to use the synthetic data path by pointing the settings
at a tmp directory with no warehouse + no labels. Live-path behaviour is
exercised by tests marked ``@pytest.mark.integration`` (which CI skips).
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _force_synthetic_data_path(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """Point DATA_DIR / DUCKDB_PATH at a clean tmp dir for unit tests.

    Otherwise local dev (where data/warehouse.duckdb exists) would route
    the API through `regime.api.live`, which has a different response shape
    than `regime.api.sample` and breaks tests that pin to the sample shape.

    Tests that explicitly want the warehouse path mark themselves with
    @pytest.mark.integration; this fixture leaves them alone.
    """
    if request.node.get_closest_marker("integration"):
        return

    tmp = tmp_path_factory.mktemp("regime_data_isolated")
    monkeypatch.setenv("DATA_DIR", str(tmp))
    monkeypatch.setenv("DUCKDB_PATH", str(tmp / "warehouse.duckdb"))
    monkeypatch.setenv("RAW_DIR", str(tmp / "raw"))

    # Clear the cached Settings so the new env vars take effect
    from regime.config import get_settings

    get_settings.cache_clear()
