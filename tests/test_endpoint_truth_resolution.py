import datetime
from src.endpoint_rules import EmptyPayloadPolicy
from src.endpoint_truth import (
    EndpointPayloadClass,
    EndpointStateRow,
    FreshnessState,
    NaReasonCode,
    PayloadAssessment,
    classify_payload,
    resolve_effective_payload
)

class MockResponse:
    def __init__(self, ok, pj, ph, err=None):
        self.ok = ok
        self.payload_json = pj
        self.payload_hash = ph
        self.error_message = err

def test_stale_too_old_cutoff():
    """
    Asserts that if age > invalid_after_seconds, the system cuts off carry forward
    and drops to ERROR + STALE_TOO_OLD.
    """
    current_ts = datetime.datetime(2025, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)
    
    assessment = PayloadAssessment(
        payload_class=EndpointPayloadClass.SUCCESS_EMPTY_VALID,
        empty_policy=EmptyPayloadPolicy.EMPTY_MEANS_STALE,
        is_empty=True,
        changed=False,
        error_reason=None
    )
    
    prev_success_ts = current_ts - datetime.timedelta(hours=3)
    prev_state = EndpointStateRow(
        last_success_event_id="prev_123",
        last_success_ts_utc=prev_success_ts,
        last_payload_hash="hash1",
        last_change_event_id="prev_123",
        last_change_ts_utc=prev_success_ts
    )
    
    resolved = resolve_effective_payload(
        current_event_id="curr_123",
        current_ts_raw=current_ts,
        assessment=assessment,
        prev_state=prev_state,
        fallback_max_age_seconds=900,
        invalid_after_seconds=3600
    )
    
    assert resolved.freshness_state == FreshnessState.ERROR
    assert resolved.used_event_id is None
    assert resolved.na_reason == NaReasonCode.STALE_TOO_OLD.value

def test_correct_stale_age_under_unchanged_payloads():
    """
    Simulates:
    Cycle N: payload changed at T0 (sets last_change_ts=T0)
    Cycles N+1..N+k: payload unchanged but still 2xx (updates last_success_ts each cycle)
    Cycle N+k+1: endpoint returns ERROR at T1
    Expected: resolved lineage for that endpoint is STALE_CARRY with stale_age_seconds approx (T1 - T0).
    """
    t0 = datetime.datetime(2025, 1, 1, 9, 30, tzinfo=datetime.timezone.utc)
    t_intermediate = datetime.datetime(2025, 1, 1, 9, 45, tzinfo=datetime.timezone.utc)
    t1 = datetime.datetime(2025, 1, 1, 10, 0, tzinfo=datetime.timezone.utc)

    # State after N+k cycles (last success is recent, but last change was T0)
    prev_state = EndpointStateRow(
        last_success_event_id="event_intermediate",
        last_success_ts_utc=t_intermediate,
        last_payload_hash="hash1",
        last_change_event_id="event_t0",
        last_change_ts_utc=t0
    )

    assessment = PayloadAssessment(
        payload_class=EndpointPayloadClass.ERROR,
        empty_policy=EmptyPayloadPolicy.EMPTY_INVALID,
        is_empty=False,
        changed=None,
        error_reason="HTTP Error"
    )

    resolved = resolve_effective_payload(
        current_event_id="event_t1",
        current_ts_raw=t1,
        assessment=assessment,
        prev_state=prev_state,
        fallback_max_age_seconds=3600,
        invalid_after_seconds=7200
    )

    assert resolved.freshness_state == FreshnessState.STALE_CARRY
    assert resolved.used_event_id == "event_t0"
    
    expected_age = int((t1 - t0).total_seconds())
    assert resolved.stale_age_seconds == expected_age
    assert resolved.na_reason == NaReasonCode.CARRY_FORWARD_ERROR.value

def test_provider_error_envelope_classification():
    error_payload = {
        "status": 500,
        "error": "Internal Server Error",
        "message": "Gateway Timeout fetching options"
    }
    
    res = MockResponse(True, error_payload, "error_hash")
    assessment = classify_payload(res, "old_hash", "GET", "/api/stock/AAPL/option-chains", "RTH")
    
    assert assessment.payload_class == EndpointPayloadClass.ERROR
    assert assessment.error_reason == "PROVIDER_ERROR_ENVELOPE"

def test_empty_means_stale_no_prior():
    current_ts = datetime.datetime.now(datetime.timezone.utc)
    assessment = PayloadAssessment(
        payload_class=EndpointPayloadClass.SUCCESS_EMPTY_VALID,
        empty_policy=EmptyPayloadPolicy.EMPTY_MEANS_STALE,
        is_empty=True,
        changed=False,
        error_reason=None
    )
    
    resolved = resolve_effective_payload(
        current_event_id="curr_123",
        current_ts_raw=current_ts,
        assessment=assessment,
        prev_state=None
    )
    
    assert resolved.freshness_state == FreshnessState.ERROR
    assert resolved.used_event_id is None
    assert resolved.na_reason == NaReasonCode.NO_PRIOR_SUCCESS.value