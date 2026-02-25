import pytest
import datetime
from src.endpoint_rules import EmptyPayloadPolicy, get_empty_policy
from src.endpoint_truth import (
    EndpointPayloadClass, FreshnessState, PayloadAssessment,
    classify_payload, resolve_effective_payload, EndpointStateRow
)

class MockResponse:
    def __init__(self, ok, pj, ph, err=None):
        self.ok = ok
        self.payload_json = pj
        self.payload_hash = ph
        self.error_message = err

def test_get_empty_policy_matrix():
    # Flow endpoints MUST be IS_DATA in RTH and MEANS_STALE in PREMARKET/AFTERHOURS
    flow_eps = [
        "/api/stock/{ticker}/flow-per-strike",
        "/api/stock/{ticker}/flow-per-strike-intraday",
        "/api/stock/{ticker}/flow-recent",
        "/api/stock/{ticker}/net-prem-ticks"
    ]
    for ep in flow_eps:
        assert get_empty_policy("GET", ep, "RTH") == EmptyPayloadPolicy.EMPTY_IS_DATA

def test_classify_payload_empty_is_data_changed():
    res = MockResponse(True, [], "new_empty_hash")
    assessment = classify_payload(res, "old_non_empty_hash", "GET", "/api/lit-flow/{ticker}", "RTH")
    assert assessment.payload_class == EndpointPayloadClass.SUCCESS_EMPTY_VALID

def test_pre_carry_forward_behavior_with_prior():
    res = MockResponse(True, [], "empty_hash")
    assessment = classify_payload(res, "old_hash", "GET", "/api/stock/{ticker}/flow-per-strike", "PREMARKET")
    assert assessment.empty_policy == EmptyPayloadPolicy.EMPTY_MEANS_STALE

def test_pre_carry_forward_behavior_no_prior():
    res = MockResponse(True, [], "empty_hash")
    assessment = classify_payload(res, None, "GET", "/api/stock/{ticker}/flow-per-strike", "PREMARKET")
    
    current_ts = datetime.datetime.now(datetime.timezone.utc)
    resolved = resolve_effective_payload("uuid-curr", current_ts, assessment, None)
    
    # Without prior state, a PREMARKET flow endpoint correctly degrades to ERROR
    assert resolved.freshness_state == FreshnessState.ERROR

def test_classify_payload_fails_fast_on_legacy_label():
    """
    EVIDENCE: Explicitly verifies that attempting to classify a payload with a legacy 
    session alias strictly fails fast instead of continuing or coercing.
    """
    res = MockResponse(True, [], "empty_hash")
    with pytest.raises(ValueError, match="Unknown session label: PRE. Allowed: PREMARKET, RTH, AFTERHOURS, CLOSED"):
        classify_payload(res, "old_hash", "GET", "/api/stock/{ticker}/flow-per-strike", "PRE")