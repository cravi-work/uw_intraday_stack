import pytest
from src.features import extract_all
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