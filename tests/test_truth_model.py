import datetime
from src.time_utils import to_utc_dt, UTC
from src.endpoint_rules import get_empty_policy, EmptyPayloadPolicy
from src.endpoint_truth import classify_payload, resolve_effective_payload, EndpointPayloadClass, FreshnessState

class MockResponse:
    def __init__(self, ok, pj, ph, err=None):
        self.ok = ok
        self.payload_json = pj
        self.payload_hash = ph
        self.error_message = err

def test_get_empty_policy_matrix():
    # 1. Flow endpoints MUST be IS_DATA in REG and MEANS_STALE in PRE/AFT
    flow_eps = [
        "/api/stock/AAPL/flow-per-strike",
        "/api/stock/AAPL/flow-per-strike-intraday",
        "/api/stock/AAPL/flow-recent",
        "/api/stock/AAPL/net-prem-ticks"
    ]
    for ep in flow_eps:
        assert get_empty_policy("GET", ep, "REG") == EmptyPayloadPolicy.EMPTY_IS_DATA
        assert get_empty_policy("GET", ep, "PRE") == EmptyPayloadPolicy.EMPTY_MEANS_STALE
        assert get_empty_policy("GET", ep, "AFT") == EmptyPayloadPolicy.EMPTY_MEANS_STALE
        assert get_empty_policy("GET", ep, "CLOSED") == EmptyPayloadPolicy.EMPTY_MEANS_STALE

    # 2. Alerts/Lit-Flow MUST be IS_DATA across all sessions
    always_eps = [
        "/api/stock/AAPL/flow-alerts",
        "/api/darkpool/AAPL",
        "/api/lit-flow/AAPL"
    ]
    for ep in always_eps:
        assert get_empty_policy("GET", ep, "REG") == EmptyPayloadPolicy.EMPTY_IS_DATA
        assert get_empty_policy("GET", ep, "PRE") == EmptyPayloadPolicy.EMPTY_IS_DATA

    # 3. Structural Endpoints MUST be INVALID (Fail Closed)
    assert get_empty_policy("GET", "/api/stock/AAPL/option-chains", "REG") == EmptyPayloadPolicy.EMPTY_INVALID

def test_classify_payload_empty_is_data_changed():
    # Empty payload + EMPTY_IS_DATA + prev_hash different => changed=True
    res = MockResponse(True, [], "new_empty_hash")
    assessment = classify_payload(res, "old_non_empty_hash", "GET", "/api/lit-flow/AAPL", "REG")
    
    assert assessment.payload_class == EndpointPayloadClass.SUCCESS_EMPTY_VALID
    assert assessment.empty_policy == EmptyPayloadPolicy.EMPTY_IS_DATA
    assert assessment.changed is True

def test_pre_carry_forward_behavior_with_prior():
    # If PRE flow endpoint returns empty and prior success exists => STALE_CARRY
    res = MockResponse(True, [], "empty_hash")
    assessment = classify_payload(res, "old_hash", "GET", "/api/stock/AAPL/flow-per-strike", "PRE")
    
    assert assessment.empty_policy == EmptyPayloadPolicy.EMPTY_MEANS_STALE
    
    current_ts = datetime.datetime(2025, 1, 1, 9, 0, tzinfo=UTC)
    prev_success_ts = datetime.datetime(2024, 12, 31, 16, 0, tzinfo=UTC)
    
    prev_state = {
        "last_success_event_id": "uuid-prev-123",
        "last_success_ts_utc": prev_success_ts,
        "last_change_event_id": "uuid-prev-123",
        "last_change_ts_utc": prev_success_ts
    }
    
    resolved = resolve_effective_payload("uuid-curr", current_ts, assessment, prev_state)
    
    assert resolved.freshness_state == FreshnessState.STALE_CARRY
    assert resolved.used_event_id == "uuid-prev-123"
    assert resolved.stale_age_seconds == int((current_ts - prev_success_ts).total_seconds())
    assert resolved.stale_age_seconds > 0
    assert resolved.na_reason == "CARRY_FORWARD_SUCCESS_EMPTY_VALID"

def test_pre_carry_forward_behavior_no_prior():
    # If no prior success exists => NA path with explicit NO_PRIOR_SUCCESS
    res = MockResponse(True, [], "empty_hash")
    assessment = classify_payload(res, None, "GET", "/api/stock/AAPL/flow-per-strike", "PRE")
    
    current_ts = datetime.datetime.now(UTC)
    prev_state = None # No prior state available to fall back on
    
    resolved = resolve_effective_payload("uuid-curr", current_ts, assessment, prev_state)
    
    assert resolved.freshness_state == FreshnessState.EMPTY_VALID
    assert resolved.used_event_id is None
    assert resolved.na_reason == "NO_PRIOR_SUCCESS"