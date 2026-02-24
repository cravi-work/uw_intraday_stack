import pytest
from src.analytics import build_gex_levels

def test_build_gex_levels_schema():
    """
    EVIDENCE: Asserts that build_gex_levels conforms to the expected Tuple structure downstream.
    Proves frozen interface stability.
    """
    payload = [
        {"strike": 100, "gamma_exposure": 5000},
        {"strike": 105, "gamma_exposure": -2000},
        {"strike": 102.5, "gamma_exposure": 100}
    ]
    
    levels = build_gex_levels(payload)
    
    assert isinstance(levels, list)
    assert len(levels) >= 2 # Should find POS_MAX, NEG_MAX, and potentially FLIP
    
    for lvl in levels:
        assert len(lvl) == 4, f"Level tuple must have 4 elements, got {len(lvl)}"
        l_type, price, mag, meta = lvl
        
        assert isinstance(l_type, str)
        assert isinstance(price, float)
        assert mag is None or isinstance(mag, float)
        assert isinstance(meta, dict)