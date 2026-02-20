import pytest
from src.features import extract_all
from src.endpoint_truth import EndpointContext

def test_no_hallucination_replay_defaults():
    """Asserts that missing payloads do not inject `0.0` but strictly pass None."""
    ctx = EndpointContext(
        endpoint_id=1, method="GET", path="/api/stock/{ticker}/ohlc/{candle_size}",
        operation_id="opt", signature="GET /api/stock/{ticker}/ohlc/1m",
        used_event_id=None, payload_class="ERROR", freshness_state="ERROR",
        stale_age_min=None, na_reason="missing_raw_payload_for_lineage"
    )
    
    effective_payloads = {1: None}
    contexts = {1: ctx}
    
    f_rows, _ = extract_all(effective_payloads, contexts)
    
    spot_row = next((f for f in f_rows if f["feature_key"] == "spot"), None)
    assert spot_row is not None
    assert spot_row["feature_value"] is None # Must explicitly be None, not 0.0
    assert spot_row["meta_json"]["na_reason"] == "missing_raw_payload_for_lineage"