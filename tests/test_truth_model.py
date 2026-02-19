import datetime
from src.time_utils import to_utc_dt, UTC
from src.endpoint_rules import is_empty_valid
from src.endpoint_truth import classify_payload, resolve_effective_payload, EndpointPayloadClass, FreshnessState

class MockResponse:
    def __init__(self, ok, pj, ph, err=None):
        self.ok = ok
        self.payload_json = pj
        self.payload_hash = ph
        self.error_message = err

def test_timestamp_normalization():
    fallback = datetime.datetime(2025, 1, 1, tzinfo=UTC)
    assert to_utc_dt(None, fallback=fallback) == fallback
    assert to_utc_dt(1700000000.0, fallback=fallback) == datetime.datetime.fromtimestamp(1700000000.0, UTC)

def test_empty_valid_policy():
    # Structural endpoint NEVER valid empty
    assert not is_empty_valid("GET", "/api/stock/AAPL/option-chains", "REG")
    assert not is_empty_valid("GET", "/api/stock/AAPL/option-chains", "PRE")
    
    # Session endpoint valid ONLY in extended
    assert not is_empty_valid("GET", "/api/stock/AAPL/flow-per-strike", "REG")
    assert is_empty_valid("GET", "/api/stock/AAPL/flow-per-strike", "AFT")

    # Transactional endpoint ALWAYS valid
    assert is_empty_valid("GET", "/api/lit-flow/AAPL", "REG")

def test_carry_forward_correctness():
    # Simulate a PRE market empty options chain (Not allowed, Should ERROR -> Carry Forward)
    res = MockResponse(True, {}, "hash1")
    pclass, _ = classify_payload(res, "old_hash", "GET", "/api/stock/AAPL/option-chains", "PRE")
    assert pclass == EndpointPayloadClass.ERROR

    prev_state = {
        "last_success_event_id": "uuid-1234",
        "last_success_ts_utc": datetime.datetime.now(UTC) - datetime.timedelta(minutes=5)
    }

    resolved = resolve_effective_payload("uuid-current", datetime.datetime.now(UTC), pclass, prev_state)
    assert resolved.freshness_state == FreshnessState.STALE_CARRY
    assert resolved.used_event_id == "uuid-1234"
    assert resolved.stale_age_seconds > 0

def test_age_clamping():
    # Simulate current_ts being slightly behind prev_ts due to clock drift
    current_ts = datetime.datetime.now(UTC)
    prev_state = {
        "last_success_event_id": "uuid-1234",
        "last_success_ts_utc": current_ts + datetime.timedelta(seconds=5)
    }
    resolved = resolve_effective_payload("uuid-current", current_ts, EndpointPayloadClass.ERROR, prev_state)
    assert resolved.stale_age_seconds == 0  # Clamped, never negative