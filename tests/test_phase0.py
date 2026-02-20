import pytest
from src.features import extract_smart_whale_pressure
from src.endpoint_truth import EndpointContext

@pytest.fixture
def dummy_ctx():
    return EndpointContext(
        endpoint_id=1, method="GET", path="/api/stock/{ticker}/flow-per-strike",
        operation_id="opt", signature="GET /api/stock/{ticker}/flow-per-strike",
        used_event_id="uuid", payload_class="SUCCESS_HAS_DATA", freshness_state="FRESH",
        stale_age_min=0, na_reason=None
    )

def test_whale_missing_payload_is_na(dummy_ctx):
    res = extract_smart_whale_pressure(None, dummy_ctx)
    assert res.features["smart_whale_pressure"] is None

def test_whale_schema_non_dict_rows_is_na(dummy_ctx):
    payload = {"data": ["error_message_1", "error_message_2"]}
    res = extract_smart_whale_pressure(payload, dummy_ctx)
    assert res.features["smart_whale_pressure"] is None

def test_whale_filtered_zero_is_zero(dummy_ctx):
    payload = [{"premium": 500, "dte": 0, "side": "BUY", "put_call": "CALL"}]
    res = extract_smart_whale_pressure(payload, dummy_ctx, min_premium=10000)
    assert res.features["smart_whale_pressure"] == 0.0

def test_whale_unparseable_is_na(dummy_ctx):
    payload = [{"dte": 0, "side": "BUY", "put_call": "CALL"}]
    res = extract_smart_whale_pressure(payload, dummy_ctx)
    assert res.features["smart_whale_pressure"] is None

def test_whale_unknown_side_labels_is_na(dummy_ctx):
    payload = [{"premium": 50000, "dte": 0, "side": "MID", "put_call": "CALL"}]
    res = extract_smart_whale_pressure(payload, dummy_ctx)
    assert res.features["smart_whale_pressure"] is None