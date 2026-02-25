import pytest
import datetime as dt
from unittest.mock import MagicMock
from src.features import (
    extract_price_features,
    extract_smart_whale_pressure,
    EndpointContext
)

@pytest.fixture
def mock_ctx():
    ctx = MagicMock(spec=EndpointContext)
    ctx.freshness_state = "FRESH"
    ctx.na_reason = None
    ctx.path = "/api/mock"
    ctx.method = "GET"
    ctx.operation_id = "test"
    ctx.used_event_id = "mock-id"
    ctx.signature = "sig"
    ctx.endpoint_id = 1
    ctx.stale_age_min = 0
    ctx.effective_ts_utc = dt.datetime(2026, 1, 1, 10, 0, 0, tzinfo=dt.timezone.utc)
    return ctx

def test_payload_valid_timestamp_priority(mock_ctx):
    """
    EVIDENCE: payload has valid timestamp -> lineage uses payload timestamp.
    Spot price explicitly parses the timestamp from the row and emits it to _build_meta.
    """
    payload = [{"close": 150.0, "t": 1700000000.0}]
    bundle = extract_price_features(payload, mock_ctx)
    
    lineage = bundle.meta["price"]["metric_lineage"]
    assert lineage["timestamp_source"] == "payload"
    assert lineage["timestamp_quality"] == "VALID"
    assert lineage["effective_ts_utc"] is not None

def test_context_timestamp_fallback(mock_ctx):
    """
    EVIDENCE: payload lacks timestamp, context has timestamp -> lineage uses context timestamp.
    Options flow does not have strict row-level UTC mapping in the extractor, so it delegates to context.
    """
    payload = [{"premium": 15000.0, "dte": 5.0, "side": "ASK", "put_call": "CALL"}]
    bundle = extract_smart_whale_pressure(payload, mock_ctx)
    
    lineage = bundle.meta["flow"]["metric_lineage"]
    assert lineage["timestamp_source"] == "endpoint_context"
    assert lineage["timestamp_quality"] == "VALID"
    assert lineage["effective_ts_utc"] == mock_ctx.effective_ts_utc.isoformat()

def test_missing_timestamp_both_sources(mock_ctx):
    """
    EVIDENCE: neither available -> lineage timestamp is missing and tagged as missing.
    """
    mock_ctx.effective_ts_utc = None
    payload = [{"premium": 15000.0, "dte": 5.0, "side": "ASK", "put_call": "CALL"}]
    bundle = extract_smart_whale_pressure(payload, mock_ctx)
    
    lineage = bundle.meta["flow"]["metric_lineage"]
    assert lineage["timestamp_source"] == "missing"
    assert lineage["timestamp_quality"] == "MISSING"
    assert lineage["effective_ts_utc"] is None

def test_malformed_payload_timestamp_keeps_feature(mock_ctx):
    """
    EVIDENCE: If payload timestamp is malformed, mark timestamp as missing/invalid, 
    keep the feature value only if feature extraction itself is valid.
    """
    payload = [{"close": 150.0, "t": "invalid_date_string"}]
    bundle = extract_price_features(payload, mock_ctx)
    
    assert bundle.features["spot"] == 150.0  # Feature value is preserved
    
    lineage = bundle.meta["price"]["metric_lineage"]
    assert lineage["timestamp_source"] == "payload"
    assert lineage["timestamp_quality"] == "INVALID"  # Flagged explicitly for gating
    assert lineage["effective_ts_utc"] is None