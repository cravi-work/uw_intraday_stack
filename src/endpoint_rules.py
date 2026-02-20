from __future__ import annotations
from enum import Enum

class EmptyPayloadPolicy(Enum):
    """
    EMPTY_IS_DATA: empty payload is a valid measurement ("zero events") and should update last_success_*
    EMPTY_MEANS_STALE: empty payload is not data (session/venue limitation); should not overwrite last success; triggers stale carry-forward
    EMPTY_INVALID: empty payload indicates provider or query error; treat as error
    """
    EMPTY_IS_DATA = "EMPTY_IS_DATA"
    EMPTY_MEANS_STALE = "EMPTY_MEANS_STALE"
    EMPTY_INVALID = "EMPTY_INVALID"

# Endpoints where empty is ALWAYS valid data (e.g., legitimately zero alerts or darkpool prints)
ALWAYS_EMPTY_IS_DATA_PATHS = {
    "/api/stock/{ticker}/flow-alerts",
    "/api/darkpool/{ticker}",
    "/api/lit-flow/{ticker}"
}

# Endpoints where empty is data in REG, but means STALE in PRE/AFT/CLOSED (Options volume feeds)
SESSION_AWARE_FLOW_PATHS = {
    "/api/stock/{ticker}/flow-per-strike",
    "/api/stock/{ticker}/flow-per-strike-intraday",
    "/api/stock/{ticker}/flow-recent",
    "/api/stock/{ticker}/net-prem-ticks"
}

# Structural Options endpoints: should NEVER be empty in REG (Error), but are STALE in PRE/AFT
SESSION_AWARE_OPTIONS_PATHS = {
    "/api/stock/{ticker}/oi-per-strike",
    "/api/stock/{ticker}/oi-change",
    "/api/stock/{ticker}/option/volume-oi-expiry",
    "/api/stock/{ticker}/option-chains",
    "/api/stock/{ticker}/option-contracts",
    "/api/stock/{ticker}/volatility/term-structure",
    "/api/stock/{ticker}/interpolated-iv",
    "/api/stock/{ticker}/iv-rank",
    "/api/stock/{ticker}/historical-risk-reversal-skew",
    "/api/stock/{ticker}/greek-exposure",
    "/api/stock/{ticker}/greek-exposure/strike",
    "/api/stock/{ticker}/greek-exposure/expiry",
    "/api/stock/{ticker}/spot-exposures",
    "/api/stock/{ticker}/spot-exposures/strike",
    "/api/stock/{ticker}/spot-exposures/expiry-strike",
    "/api/stock/{ticker}/max-pain"
}

def get_empty_policy(method: str, path: str, session_label: str) -> EmptyPayloadPolicy:
    """
    Determines if an empty JSON payload ({} or []) is legitimate data, 
    stale data (due to extended hours), or an invalid error state based on exact path mapping.
    """
    if method.upper() != "GET":
        return EmptyPayloadPolicy.EMPTY_INVALID

    if path in ALWAYS_EMPTY_IS_DATA_PATHS:
        return EmptyPayloadPolicy.EMPTY_IS_DATA

    if path in SESSION_AWARE_FLOW_PATHS:
        if session_label in ("PRE", "AFT", "CLOSED"):
            return EmptyPayloadPolicy.EMPTY_MEANS_STALE
        return EmptyPayloadPolicy.EMPTY_IS_DATA

    if path in SESSION_AWARE_OPTIONS_PATHS:
        if session_label in ("PRE", "AFT", "CLOSED"):
            return EmptyPayloadPolicy.EMPTY_MEANS_STALE
        # During REG, these structural endpoints must fail-closed if empty
        return EmptyPayloadPolicy.EMPTY_INVALID

    # Fail closed: If not explicitly handled, an empty payload is an error.
    return EmptyPayloadPolicy.EMPTY_INVALID