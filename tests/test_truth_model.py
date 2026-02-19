import datetime
from src.endpoint_truth import to_utc_dt, classify_payload, resolve_effective_payload, EndpointPayloadClass, FreshnessState
from src.endpoint_rules import get_empty_policy, EmptyPayloadPolicy

UTC = datetime.timezone.utc

class MockResponse:
    def __init__(self, ok, pj, ph, err=None):
        self.ok = ok
        self.payload_json = pj
        self.payload_hash = ph
        self.error_message = err

def test_timestamp_normalization():
    fallback = datetime.datetime(2025, 1, 1, tzinfo=UTC)
    assert to_utc_dt(None, fallback=fallback) == fallback
    # Should convert numeric epoch to aware UTC datetime
    assert to_utc_dt(1700000000.0, fallback=fallback) == datetime.datetime.fromtimestamp(1700000000.0, UTC)
    # Should convert naive datetime to aware UTC datetime
    naive = datetime.datetime(2025, 1, 1)
    assert to_utc_dt(naive, fallback=fallback).tzinfo == UTC

def test_empty_valid_policy():
    # Structural endpoint NEVER valid empty
    assert get_empty_policy("GET", "/api/stock/AAPL/option-chains", "REG") == EmptyPayloadPolicy.EMPTY_INVALID
    
    # Session endpoint valid ONLY in extended
    assert get_empty_policy("GET", "/api/stock/AAPL/flow-per-strike", "REG") == EmptyPayloadPolicy.EMPTY_INVALID
    assert get_empty_policy("GET", "/api/stock/AAPL/flow-per-strike", "AFT") == EmptyPayloadPolicy.EMPTY_MEANS_STALE

    # Transactional endpoint ALWAYS valid
    assert get_empty_policy("GET", "/api/lit-flow/AAPL", "REG") == EmptyPayloadPolicy.EMPTY_IS_DATA

def test_carry_forward_correctness():
    # Empty options chain in PRE market -> ERROR (Not allowed by policy)
    res = MockResponse(True, {}, "hash1")
    assessment = classify_payload(res, "old_hash", "GET", "/api/stock/AAPL/option-chains", "PRE")
    assert assessment.payload_class == EndpointPayloadClass.ERROR

    prev_state = {
        "last_success_event_id": "uuid-1234",
        "last_success_ts_utc": datetime.datetime.now(UTC) - datetime.timedelta(minutes=5)
    }

    # Should fall back to prior state
    resolved = resolve_effective_payload("uuid-current", datetime.datetime.now(UTC), assessment, prev_state)
    assert resolved.freshness_state == FreshnessState.STALE_CARRY
    assert resolved.used_event_id == "uuid-1234"
    assert resolved.stale_age_seconds is not None and resolved.stale_age_seconds > 0

def test_age_clamping():
    current_ts = datetime.datetime.now(UTC)
    # Simulate DB having a timestamp *ahead* of current_ts (clock drift)
    prev_state = {
        "last_success_event_id": "uuid-1234",
        "last_success_ts_utc": current_ts + datetime.timedelta(seconds=5)
    }
    
    res = MockResponse(False, None, None) # Forces error state -> triggers carry forward
    assessment = classify_payload(res, None, "GET", "/api/lit-flow/AAPL", "REG")
    resolved = resolve_effective_payload("uuid-current", current_ts, assessment, prev_state)
    
    # Age must never be negative
    assert resolved.stale_age_seconds == 0