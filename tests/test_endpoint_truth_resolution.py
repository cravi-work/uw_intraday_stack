import datetime
from src.endpoint_rules import EmptyPayloadPolicy
from src.endpoint_truth import (
    EndpointPayloadClass,
    FreshnessState,
    NaReasonCode,
    PayloadAssessment,
    resolve_effective_payload
)

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

def test_empty_means_stale_with_prior():
    """
    Case B: EMPTY_MEANS_STALE + prior success exists
    Must return STALE_CARRY with the exact old event ID and computed age.
    """
    current_ts = datetime.datetime.now(datetime.timezone.utc)
    prev_success_ts = current_ts - datetime.timedelta(seconds=120)
    
    assessment = PayloadAssessment(
        payload_class=EndpointPayloadClass.SUCCESS_EMPTY_VALID,
        empty_policy=EmptyPayloadPolicy.EMPTY_MEANS_STALE,
        is_empty=True,
        changed=False,
        error_reason=None
    )
    
    prev_state = {
        "last_success_event_id": "prev_123",
        "last_success_ts_utc": prev_success_ts,
        "last_change_event_id": "prev_change_123",
        "last_change_ts_utc": prev_success_ts
    }
    
    resolved = resolve_effective_payload(
        current_event_id="curr_123",
        current_ts_raw=current_ts,
        assessment=assessment,
        prev_state=prev_state
    )
    
    assert resolved.freshness_state == FreshnessState.STALE_CARRY
    assert resolved.used_event_id == "prev_123"
    assert resolved.stale_age_seconds == 120
    assert resolved.na_reason == NaReasonCode.CARRY_FORWARD_EMPTY_MEANS_STALE.value

def test_stale_carry_invariant_coercion():
    """
    Ensures that if the truth model ever attempts a STALE_CARRY resolution 
    without a valid used_event_id, it is safely coerced into an ERROR state.
    """
    current_ts = datetime.datetime.now(datetime.timezone.utc)
    assessment = PayloadAssessment(
        payload_class=EndpointPayloadClass.SUCCESS_STALE,
        empty_policy=EmptyPayloadPolicy.EMPTY_MEANS_STALE,
        is_empty=False,
        changed=False,
        error_reason=None
    )
    
    # Simulate DB state returning a success_ts but somehow missing the event_id
    prev_state_corrupt = {
        "last_change_event_id": None,
        "last_change_ts_utc": current_ts - datetime.timedelta(seconds=120),
        "last_success_event_id": None,
        "last_success_ts_utc": current_ts - datetime.timedelta(seconds=120)
    }
    
    resolved = resolve_effective_payload(
        current_event_id=None, # e.g. current ID missing
        current_ts_raw=current_ts,
        assessment=assessment,
        prev_state=prev_state_corrupt
    )
    
    assert resolved.freshness_state == FreshnessState.ERROR
    assert resolved.used_event_id is None
    assert resolved.na_reason == NaReasonCode.NO_PRIOR_SUCCESS.value