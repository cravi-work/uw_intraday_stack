from __future__ import annotations

def is_empty_valid(method: str, path: str, session_label: str) -> bool:
    """
    Determines if an empty JSON payload ({} or []) is legitimate.
    Many Unusual Whales endpoints (like darkpool or alerts) naturally 
    return empty arrays when there is zero flow.
    """
    # For Phase 3, we default to treating empty 200 OKs as valid "no flow" states, 
    # but explicitly prevent them from overwriting last_success states.
    return True