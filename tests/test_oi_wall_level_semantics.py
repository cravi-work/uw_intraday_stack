import pytest
from src.analytics import build_oi_walls

def test_directional_oi_walls():
    """
    EVIDENCE: valid call/put OI rows + spot -> emits CALL_WALL and PUT_WALL correctly.
    """
    payload = [
        {"strike": 140, "open_interest": 5000, "put_call": "PUT"},
        {"strike": 145, "open_interest": 8000, "put_call": "PUT"},  # True PUT_WALL (below spot)
        {"strike": 155, "open_interest": 7000, "put_call": "CALL"}, # True CALL_WALL (above spot)
        {"strike": 160, "open_interest": 6000, "put_call": "CALL"}
    ]
    
    levels = build_oi_walls(payload, spot=150.0)
    
    assert len(levels) == 2
    call_wall = next((l for l in levels if l[0] == "CALL_WALL"), None)
    put_wall = next((l for l in levels if l[0] == "PUT_WALL"), None)
    
    assert call_wall is not None
    assert call_wall[1] == 155.0
    assert call_wall[2] == 7000.0
    assert call_wall[3]["spot_reference"] == 150.0
    
    assert put_wall is not None
    assert put_wall[1] == 145.0
    assert put_wall[2] == 8000.0
    assert put_wall[3]["spot_reference"] == 150.0

def test_missing_put_call_fallback():
    """
    EVIDENCE: missing put_call -> no directional walls (explicit generic fallback emitted).
    """
    payload = [
        {"strike": 140, "open_interest": 5000},
        {"strike": 145, "open_interest": 8000}
    ]
    
    levels = build_oi_walls(payload, spot=150.0)
    
    assert len(levels) == 1
    assert levels[0][0] == "OI_GENERIC_WALL"
    assert levels[0][1] == 145.0
    assert levels[0][3]["degraded_reason"] == "missing_put_call"

def test_missing_spot_fallback():
    """
    EVIDENCE: missing spot -> suppressed directional walls with reason tags.
    """
    payload = [
        {"strike": 140, "open_interest": 5000, "put_call": "PUT"},
        {"strike": 145, "open_interest": 8000, "put_call": "CALL"}
    ]
    
    levels = build_oi_walls(payload)
    
    assert len(levels) == 1
    assert levels[0][0] == "OI_GENERIC_WALL"
    assert levels[0][1] == 145.0
    assert levels[0][3]["degraded_reason"] == "missing_spot"

def test_spot_inferred_from_payload():
    """
    EVIDENCE: Spot can be correctly inferred from the row payload if not provided explicitly.
    """
    payload = [
        {"strike": 140, "open_interest": 5000, "put_call": "PUT", "spot": 150.0},
        {"strike": 160, "open_interest": 8000, "put_call": "CALL", "spot": 150.0}
    ]
    
    levels = build_oi_walls(payload)
    
    assert len(levels) == 2
    assert any(l[0] == "CALL_WALL" for l in levels)
    assert any(l[0] == "PUT_WALL" for l in levels)

def test_determinism():
    """
    EVIDENCE: Verify deterministic outputs for repeated runs.
    """
    payload = [
        {"strike": 140, "open_interest": 5000, "put_call": "PUT"},
        {"strike": 145, "open_interest": 8000, "put_call": "PUT"},
        {"strike": 155, "open_interest": 7000, "put_call": "CALL"}
    ]
    
    run1 = build_oi_walls(payload, spot=150.0)
    run2 = build_oi_walls(payload, spot=150.0)
    assert run1 == run2