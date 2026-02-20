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
    # Flow endpoints MUST be IS_DATA in REG and MEANS_STALE in PRE/AFT
    flow_eps = [
        "/api/stock/{ticker}/flow-per-strike",
        "/api/stock/{ticker}/flow-per-strike-intraday",
        "/api/stock/{ticker}/flow-recent",
        "/api/stock/{ticker}/net-prem-ticks"
    ]
    for ep in flow_eps:
        assert get_empty_policy("GET", ep, "REG") == EmptyPayloadPolicy.EMPTY_IS_DATA

def test_classify_payload_empty_is_data_changed():
    res = MockResponse(True, [], "new_empty_hash")
    assessment = classify_payload(res, "old_non_empty_hash", "GET", "/api/lit-flow/{ticker}", "REG")
    assert assessment.payload_class == EndpointPayloadClass.SUCCESS_EMPTY_VALID

def test_pre_carry_forward_behavior_with_prior():
    res = MockResponse(True, [], "empty_hash")
    assessment = classify_payload(res, "old_hash", "GET", "/api/stock/{ticker}/flow-per-strike", "PRE")
    assert assessment.empty_policy == EmptyPayloadPolicy.EMPTY_MEANS_STALE

def test_pre_carry_forward_behavior_no_prior():
    res = MockResponse(True, [], "empty_hash")
    assessment = classify_payload(res, None, "GET", "/api/stock/{ticker}/flow-per-strike", "PRE")
    
    current_ts = datetime.datetime.now(datetime.timezone.utc)
    resolved = resolve_effective_payload("uuid-curr", current_ts, assessment, None)
    
    # Without prior state, a PRE flow endpoint correctly degrades to ERROR
    assert resolved.freshness_state == FreshnessState.ERROR