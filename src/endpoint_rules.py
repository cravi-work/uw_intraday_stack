from __future__ import annotations
from enum import Enum

class EmptyPayloadPolicy(Enum):
    EMPTY_IS_DATA = "EMPTY_IS_DATA"
    EMPTY_MEANS_STALE = "EMPTY_MEANS_STALE"
    EMPTY_INVALID = "EMPTY_INVALID"

def get_empty_policy(method: str, path: str, session_label: str) -> EmptyPayloadPolicy:
    """
    Determines if an empty JSON payload ({} or []) is legitimate data, 
    stale data (due to extended hours), or an invalid error state.
    """
    method = method.upper()
    
    # 1. ALWAYS_EMPTY_VALID: Transactional feeds that are empty when volume is zero
    always_valid = [
        "/flow-alerts",
        "/darkpool",
        "/lit-flow"
    ]
    if any(ep in path for ep in always_valid):
        return EmptyPayloadPolicy.EMPTY_IS_DATA

    # 2. SESSION_EMPTY_VALID: Options volume feeds empty outside regular hours
    session_valid = [
        "/flow-per-strike",
        "/flow-recent",
        "/net-prem-ticks"
    ]
    if any(ep in path for ep in session_valid):
        if session_label in ("PRE", "AFT", "CLOSED"):
            return EmptyPayloadPolicy.EMPTY_MEANS_STALE
        return EmptyPayloadPolicy.EMPTY_INVALID

    # 3. NEVER_EMPTY_VALID: Structural endpoints (OHLC, chains, greeks) 
    # Must never be empty. If they are, it is a data provider error.
    return EmptyPayloadPolicy.EMPTY_INVALID