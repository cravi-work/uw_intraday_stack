import pytest
from src.features import (
    extract_all, 
    extract_price_features, 
    extract_dealer_greeks, 
    extract_smart_whale_pressure
)
from src.endpoint_truth import EndpointContext

def test_extractor_coverage_enforcement():
    """Asserts that planned endpoints missing from the registry or presence list explicitly crash ingestion."""
    ctx = EndpointContext(
        endpoint_id=1, method="GET", path="/api/stock/{ticker}/fake-unmapped-endpoint",
        operation_id="opt", signature="GET /api/stock/{ticker}/fake-unmapped-endpoint",
        used_event_id=None, payload_class="SUCCESS_HAS_DATA", freshness_state="FRESH",
        stale_age_min=None, na_reason=None
    )
    
    effective_payloads = {1: {"data": []}}
    contexts = {1: ctx}
    
    with pytest.raises(RuntimeError) as exc_info:
        extract_all(effective_payloads, contexts)
        
    assert "CRITICAL EXTRACTOR COVERAGE GAP" in str(exc_info.value)
    assert "/api/stock/{ticker}/fake-unmapped-endpoint" in str(exc_info.value)

def test_canonical_session_applicability_tokens():
    """
    EVIDENCE: Ensure no alias tokens ('PRE', 'AFT') are used in session_applicability.
    Must strictly use canonical tokens: 'PREMARKET', 'RTH', 'AFTERHOURS'.
    This protects downstream UI/reporting consumers from ambiguous mappings.
    """
    ctx = EndpointContext(
        endpoint_id=1, method="GET", path="/api/stock/{ticker}/mock-endpoint",
        operation_id="opt", signature="GET /api/stock/{ticker}/mock-endpoint",
        used_event_id=None, payload_class="SUCCESS_HAS_DATA", freshness_state="FRESH",
        stale_age_min=None, na_reason=None
    )
    
    # 1. Test Price Extractor (Should be PREMARKET/RTH/AFTERHOURS)
    f_bundle_price = extract_price_features({"data": []}, ctx)
    price_meta = f_bundle_price.meta["price"]
    price_sessions = price_meta["metric_lineage"]["session_applicability"].split("/")
    assert "PRE" not in price_sessions
    assert "AFT" not in price_sessions
    assert set(price_sessions) == {"PREMARKET", "RTH", "AFTERHOURS"}

    # 2. Test Greeks Extractor (Should be PREMARKET/RTH)
    f_bundle_greeks = extract_dealer_greeks([], ctx)
    greek_meta = f_bundle_greeks.meta["greeks"]
    greek_sessions = greek_meta["metric_lineage"]["session_applicability"].split("/")
    assert "PRE" not in greek_sessions
    assert set(greek_sessions) == {"PREMARKET", "RTH"}
    
    # 3. Test Flow Extractor (Should be strictly RTH)
    f_bundle_flow = extract_smart_whale_pressure([], ctx)
    flow_meta = f_bundle_flow.meta["flow"]
    flow_sessions = flow_meta["metric_lineage"]["session_applicability"].split("/")
    assert set(flow_sessions) == {"RTH"}