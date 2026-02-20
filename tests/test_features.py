from src.features import extract_all
from src.endpoint_truth import EndpointContext

def test_extract_all_import_contract():
    """Asserts the main orchestration boundary exists and runs without ImportError."""
    assert callable(extract_all)

def test_extract_all_na_propagation():
    """Asserts that missing payloads definitively return valid rows with None values and ERROR freshness."""
    ctx = EndpointContext(
        endpoint_id=1, method="GET", path="/api/stock/{ticker}/spot-exposures",
        operation_id="opt", signature="GET /api/stock/{ticker}/spot-exposures",
        used_event_id=None, payload_class="ERROR", freshness_state="ERROR",
        stale_age_min=None, na_reason="NO_PRIOR_SUCCESS"
    )
    
    effective_payloads = {1: None}
    contexts = {1: ctx}
    
    f_rows, l_rows = extract_all(effective_payloads, contexts)
    
    # Must yield the placeholder NA row (never 0.0)
    assert len(f_rows) > 0
    for f in f_rows:
        assert f["feature_value"] is None
        assert f["meta_json"]["freshness_state"] == "ERROR"
        assert f["meta_json"]["na_reason"] == "NO_PRIOR_SUCCESS"

def test_deterministic_sorting():
    """Asserts that the best available freshness state is preferred when multiple identical endpoints exist."""
    ctx_stale = EndpointContext(
        endpoint_id=1, method="GET", path="/api/stock/{ticker}/greek-exposure",
        operation_id="opt", signature="GET /api/stock/{ticker}/greek-exposure",
        used_event_id="old_uuid", payload_class="SUCCESS_STALE", freshness_state="STALE_CARRY",
        stale_age_min=30, na_reason="CARRY_FORWARD_EMPTY_MEANS_STALE"
    )
    
    ctx_fresh = EndpointContext(
        endpoint_id=2, method="GET", path="/api/stock/{ticker}/greek-exposure",
        operation_id="opt", signature="GET /api/stock/{ticker}/greek-exposure",
        used_event_id="new_uuid", payload_class="SUCCESS_HAS_DATA", freshness_state="FRESH",
        stale_age_min=0, na_reason=None
    )
    
    effective_payloads = {1: None, 2: []} # Valid empty list
    contexts = {1: ctx_stale, 2: ctx_fresh}
    
    f_rows, _ = extract_all(effective_payloads, contexts)
    
    # Assert it picked the FRESH context
    assert len(f_rows) > 0
    assert f_rows[0]["meta_json"]["freshness_state"] == "FRESH"
    assert f_rows[0]["meta_json"]["source_endpoints"][0]["endpoint_id"] == 2