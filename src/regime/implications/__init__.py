"""Regime implications — translate regime labels into PM-facing guidance.

This is the layer between the raw model output (a state vector and class
label) and a recommendation a portfolio manager or CIO would act on.

Public surface:
    - models: Pydantic response shapes (RegimeImplications, AllocationTilt, ...)
    - service.get_latest_implications(): the function the Streamlit page + API call
    - policy.ALLOCATION_BY_REGIME: the hand-coded tilt map
"""

from regime.implications.models import (
    AllocationTilt,
    AlternativeScenario,
    HistoricalRegimeStats,
    RegimeImplications,
)
from regime.implications.policy import (
    ALLOCATION_BY_REGIME,
    REGIME_DESCRIPTIONS,
    RISK_PROFILE,
)
from regime.implications.service import (
    get_implications_for_date,
    get_latest_implications,
)

__all__ = [
    "ALLOCATION_BY_REGIME",
    "AllocationTilt",
    "AlternativeScenario",
    "HistoricalRegimeStats",
    "REGIME_DESCRIPTIONS",
    "RISK_PROFILE",
    "RegimeImplications",
    "get_implications_for_date",
    "get_latest_implications",
]
