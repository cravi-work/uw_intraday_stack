import datetime
from src.endpoint_rules import EmptyPayloadPolicy
from src.endpoint_truth import (
    EndpointPayloadClass,
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
    
    # Simulate a prior success from 3 hours ago (10,800 seconds)
    prev_success_ts = current_ts - datetime.timedelta(hours=3)
    prev_state = {
        "last_success_event_id": "prev_123",
        "last_success_ts_utc": prev_success_ts,
        "last_change_event_id": "prev_123",
        "last_change_ts_utc": prev_success_ts
    }
    
    resolved = resolve_effective_payload(
        current_event_id="curr_123",
        current_ts_raw=current_ts,
        assessment=assessment,
        prev_state=prev_state,
        fallback_max_age_seconds=900,
        invalid_after_seconds=3600  # Hard cutoff at 1 hour
    )
    
    assert resolved.freshness_state == FreshnessState.ERROR
    assert resolved.used_event_id is None
    assert resolved.na_reason == NaReasonCode.STALE_TOO_OLD.value

def test_provider_error_envelope_classification():
    """
    Asserts that a validly formatted JSON containing only provider error metadata
    is classified as an ERROR and not as SUCCESS_HAS_DATA.
    """
    # This is valid JSON, but functionally it is an error wrapper.
    error_payload = {
        "status": 500,
        "error": "Internal Server Error",
        "message": "Gateway Timeout fetching options"
    }
    
    res = MockResponse(True, error_payload, "error_hash")
    assessment = classify_payload(res, "old_hash", "GET", "/api/stock/AAPL/option-chains", "REG")
    
    assert assessment.payload_class == EndpointPayloadClass.ERROR
    assert assessment.error_reason == "PROVIDER_ERROR_ENVELOPE"

def test_empty_means_stale_no_prior():
    """
    Case A: EMPTY_MEANS_STALE + prev_state=None 
    Must strictly return ERROR and NO_PRIOR_SUCCESS (never EMPTY_VALID).
    """
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