from src.analytics import build_gex_levels

def test_gex_flip_no_placeholder_magnitude():
    """Asserts the GEX_FLIP level outputs None for magnitude rather than a silent zero."""
    payload = [
        {"strike": 100, "gamma_exposure": -500},
        {"strike": 105, "gamma_exposure": -50},
        {"strike": 110, "gamma_exposure": 300},
        {"strike": 115, "gamma_exposure": 700}
    ]
    
    levels = build_gex_levels(payload)
    assert len(levels) == 3 # NEG_MAX, POS_MAX, FLIP
    
    flip_level = next(l for l in levels if l[0] == "GEX_FLIP")
    assert flip_level[1] == 105 # Price closest to zero
    assert flip_level[2] is None # Strict NA magnitude requirement

def test_level_meta_contract_shape():
    """Asserts details metadata outputs properly formatted input/parsed row counts."""
    payload = [{"strike": 100, "gamma_exposure": 500}]
    levels = build_gex_levels(payload)
    
    assert len(levels) == 2 # POS_MAX, FLIP
    pos_level = next(l for l in levels if l[0] == "GEX_POS_MAX")
    
    details = pos_level[3]
    assert isinstance(details, dict)
    assert details["input_rows"] == 1
    assert details["parsed_rows"] == 1
    assert "raw" not in details # Raw dump explicitly disallowed