from __future__ import annotations

def is_empty_valid(method: str, path: str, session_label: str) -> bool:
    """
    Determines if an empty JSON payload ({} or []) is legitimately empty 
    based on the specific endpoint and the market session.
    """
    method = method.upper()
    
    # 1. ALWAYS_EMPTY_VALID: Transactional feeds that are empty when volume is zero
    always_valid = [
        "/flow-alerts",
        "/darkpool",
        "/lit-flow"
    ]
    if any(ep in path for ep in always_valid):
        return True

    # 2. SESSION_EMPTY_VALID: Options volume feeds empty outside regular hours
    session_valid = [
        "/flow-per-strike",
        "/flow-recent",
        "/net-prem-ticks"
    ]
    if any(ep in path for ep in session_valid):
        if session_label in ("PRE", "AFT", "CLOSED"):
            return True
        return False

    # 3. NEVER_EMPTY_VALID: Structural endpoints (OHLC, chains, greeks) 
    # Must never be empty. If they are, it is a data provider error.
    return False