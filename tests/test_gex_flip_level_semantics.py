import pytest
from src.analytics import build_gex_levels

def test_gex_flip_sign_crossing_interpolation():
    """
    EVIDENCE: Clear sign change across strikes -> GEX_FLIP emitted at expected interpolated level.
    """
    payload = [
        {"strike": 100, "gamma_exposure": -5000},
        {"strike": 110, "gamma_exposure": 5000},
        {"strike": 120, "gamma_exposure": 10000}
    ]
    
    levels = build_gex_levels(payload)
    
    flip_level = next((l for l in levels if l[0] == "GEX_FLIP"), None)
    assert flip_level is not None
    
    # Linear interpolation between (-5000 at 100) and (5000 at 110) should yield exactly 105
    assert flip_level[1] == 105.0
    assert flip_level[3]["method"] == "sign_crossing_interpolation"
    assert flip_level[3]["p1_strike"] == 100
    assert flip_level[3]["p2_strike"] == 110

def test_gex_flip_no_sign_change():
    """
    EVIDENCE: No sign change -> no GEX_FLIP emitted. Do not fabricate a flip when all have the same sign.
    """
    payload = [
        {"strike": 100, "gamma_exposure": 5000},
        {"strike": 110, "gamma_exposure": 15000},
        {"strike": 120, "gamma_exposure": 10000}
    ]
    
    levels = build_gex_levels(payload)
    
    flip_level = next((l for l in levels if l[0] == "GEX_FLIP"), None)
    assert flip_level is None

def test_gex_flip_malformed_rows():
    """
    EVIDENCE: Malformed rows mixed in -> robust parsing skips them and maintains deterministic output.
    """
    payload = [
        {"strike": 100, "gamma_exposure": -2000},
        {"strike": "BAD_STRIKE", "gamma_exposure": 0},
        {"strike": 105, "gamma": float('nan')},
        {"strike": 110, "gamma_exposure": 8000}
    ]
    
    levels = build_gex_levels(payload)
    
    flip_level = next((l for l in levels if l[0] == "GEX_FLIP"), None)
    assert flip_level is not None
    
    # Distance: 10 strikes, GEX Range: 10000. 
    # Formula: 100 - (-2000 * (10 / 10000)) = 100 + 2 = 102.0
    assert flip_level[1] == 102.0
    
    # Assert determinism on repeat parsing
    levels_repeat = build_gex_levels(payload)
    assert levels == levels_repeat