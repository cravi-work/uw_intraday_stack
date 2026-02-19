import pytest
import math
from src.na import grab_list
from src.features import extract_smart_whale_pressure
from src.models import bounded_additive_score

# --- 1. NA Layer Tests ---

def test_grab_list_filters_non_dicts():
    """Critical: grab_list must NEVER return strings/ints/Nones."""
    # Case A: Mixed list
    payload = {"data": [{"valid": 1}, "broken", None, 123, []]}
    res = grab_list(payload)
    assert len(res) == 1
    assert res[0] == {"valid": 1}

    # Case B: Pure garbage list
    payload_garbage = ["string", "string"]
    res_garbage = grab_list(payload_garbage)
    assert res_garbage == [] 

def test_grab_list_handles_alternate_wrappers():
    """Enhancement: Verify logic works for 'trades' and 'results' keys too."""
    # Case A: trades wrapper
    p1 = {"trades": [{"a": 1}, "bad"]}
    assert len(grab_list(p1)) == 1
    
    # Case B: results wrapper
    p2 = {"results": [{"b": 2}, None]}
    assert len(grab_list(p2)) == 1

# --- 2. Whale Extractor Tests ---

def test_whale_missing_payload_is_na():
    """Missing/Null payload must return None (Unknown)."""
    res = extract_smart_whale_pressure(None)
    assert res.features["smart_whale_pressure"] is None
    assert res.meta["flow"]["na_reason"] == "missing_payload"

def test_whale_schema_non_dict_rows_is_na():
    """Schema break (list of strings) must return None."""
    payload = {"data": ["error_message_1", "error_message_2"]}
    res = extract_smart_whale_pressure(payload)
    assert res.features["smart_whale_pressure"] is None
    assert res.meta["flow"]["na_reason"] == "schema_non_dict_rows"

def test_whale_filtered_zero_is_zero():
    """Valid trades filtered by policy must return 0.0 (Valid Signal)."""
    # 1 valid trade, but premium (500) < min_premium (10000)
    payload = [{"premium": 500, "dte": 0, "side": "BUY", "put_call": "CALL"}]
    res = extract_smart_whale_pressure(payload, min_premium=10000)
    
    assert res.features["smart_whale_pressure"] == 0.0
    assert res.meta["flow"]["status"] == "filtered_zero"
    # Verify we tracked it as parseable (policy filter, not schema break)
    assert res.meta["flow"]["parseable"] == 1 

def test_whale_unparseable_is_na():
    """Missing required fields must return None."""
    # Trade missing 'premium' field
    payload = [{"dte": 0, "side": "BUY", "put_call": "CALL"}] 
    res = extract_smart_whale_pressure(payload)
    assert res.features["smart_whale_pressure"] is None
    assert "missing_required_fields" in res.meta["flow"]["na_reason"]

def test_whale_unknown_side_labels_is_na():
    """Enhancement: Semantic drift (e.g. 'MID') must return None, not 0.0."""
    # 'MID' is not in [BUY, SELL, ASK, BID, BULLISH, BEARISH]
    payload = [{"premium": 50000, "dte": 0, "side": "MID", "put_call": "CALL"}]
    res = extract_smart_whale_pressure(payload)
    
    assert res.features["smart_whale_pressure"] is None
    assert "unrecognized_side_labels" in res.meta["flow"]["na_reason"]

# --- 3. Model Logic Tests ---

def test_direction_margin_changes_bias():
    """Prove direction_margin is active."""
    features = {"f1": 0.18}
    weights = {"f1": 1.0}
    
    # Case A: No margin -> Bullish
    pred_a = bounded_additive_score(
        features, 1.0, weights, neutral_threshold=0.15, direction_margin=0.0
    )
    assert pred_a.prob_up > pred_a.prob_down
    
    # Case B: With Margin -> Neutral
    pred_b = bounded_additive_score(
        features, 1.0, weights, neutral_threshold=0.15, direction_margin=0.05
    )
    assert pred_b.prob_up == pred_b.prob_down

def test_model_hash_changes_when_params_change():
    """Audit: changing config params must change model hash."""
    features = {"f1": 1.0}
    weights = {"f1": 1.0}
    
    pred_1 = bounded_additive_score(features, 1.0, weights, neutral_threshold=0.15)
    pred_2 = bounded_additive_score(features, 1.0, weights, neutral_threshold=0.25)
    
    assert pred_1.model_hash != pred_2.model_hash

def test_prob_sum_is_unity():
    """Math Safety: Probabilities must sum to exactly 1.0."""
    features = {"f1": 0.55555}
    weights = {"f1": 1.0}
    pred = bounded_additive_score(features, 1.0, weights)
    total = pred.prob_up + pred.prob_down + pred.prob_flat
    assert total == 1.0

def test_low_coverage_forces_neutral():
    """Enhancement: Low feature coverage must kill confidence."""
    # 5 weights, but only 1 feature present (20% coverage)
    weights = {"f1": 1.0, "f2": 1.0, "f3": 1.0, "f4": 1.0, "f5": 1.0}
    features = {"f1": 1.0} # Missing f2..f5
    
    # Coverage 0.2 < 0.4 threshold -> Confidence should be 0.0
    pred = bounded_additive_score(features, data_quality_score=1.0, weights=weights)
    
    assert pred.meta["coverage"] == 0.2
    assert pred.confidence == 0.0