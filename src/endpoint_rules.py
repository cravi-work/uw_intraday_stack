from __future__ import annotations

def is_empty_valid(method: str, path: str, session_label: str) -> bool:
    """
    Determines if an empty JSON payload ({} or []) is legitimate.
    """
    # Event/transactional feeds are legitimately empty when there is zero flow
    transactional_endpoints = [
        "/flow-alerts",
        "/darkpool/",
        "/lit-flow/"
    ]
    if any(ep in path for ep in transactional_endpoints):
        return True
        
    # During Pre-market (PRE) and After-hours (AFT), options volume metrics 
    # might legitimately be empty because options do not trade in extended hours.
    if session_label in ("PRE", "AFT"):
        options_volume_endpoints = [
            "/flow-per-strike",
            "/flow-recent",
            "/net-prem-ticks"
        ]
        if any(ep in path for ep in options_volume_endpoints):
            return True

    # Base case: Structural endpoints (chains, contracts, greeks) should never be empty
    return False