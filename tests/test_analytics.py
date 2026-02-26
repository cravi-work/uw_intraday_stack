from src.analytics import build_gex_levels

def test_gex_flip_no_placeholder_magnitude():
    payload = [
        {"strike": 100, "gamma_exposure": -500},
        {"strike": 105, "gamma_exposure": -50},
        {"strike": 110, "gamma_exposure": 300},
        {"strike": 115, "gamma_exposure": 700}
    ]
    
    levels = build_gex_levels(payload)
    assert len(levels) == 3 
    
    flip_level = next(l for l in levels if l[0] == "GEX_FLIP")
    assert round(flip_level[1], 4) == 105.7143 
    assert flip_level[2] is None 

def test_level_meta_contract_shape():
    payload = [{"strike": 100, "gamma_exposure": 500}]
    levels = build_gex_levels(payload)
    
    assert len(levels) == 1 
    pos_level = levels[0]
    
    details = pos_level[3]
    assert isinstance(details, dict)
    assert details["input_rows"] == 1
    assert details["parsed_rows"] == 1
    assert "raw" not in details