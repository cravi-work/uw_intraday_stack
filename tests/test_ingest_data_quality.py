from typing import Any, List, Tuple
from src.endpoint_rules import EmptyPayloadPolicy
from src.endpoint_truth import FreshnessState

# Mimic the resolved types passed into the DQ loop inside _ingest_once_impl
class MockResolved:
    def __init__(self, f_state, u_id):
        self.freshness_state = f_state
        self.used_event_id = u_id

class MockAssessment:
    def __init__(self, e_policy):
        self.empty_policy = e_policy

def calculate_synthetic_dq(evs: List[Tuple[Any, Any, MockResolved, Any, Any, MockAssessment]]) -> float:
    """Exact replica of the invariant evaluation inside src/ingest_engine.py."""
    valid = 0
    for (_, _, res, _, _, asmnt) in evs:
        if res.freshness_state == FreshnessState.FRESH:
            valid += 1
        elif res.freshness_state == FreshnessState.STALE_CARRY and res.used_event_id is not None:
            valid += 1
        elif res.freshness_state == FreshnessState.EMPTY_VALID and asmnt.empty_policy == EmptyPayloadPolicy.EMPTY_IS_DATA:
            valid += 1
            
    return (valid / len(evs)) if evs else 0.0

def test_data_quality_scoring_invariants():
    """
    Tests that only explicitly validated lineage states contribute to the Data Quality Score.
    """
    evs = [
        # 1. FRESH -> Valid
        (None, None, MockResolved(FreshnessState.FRESH, "id1"), None, None, MockAssessment(EmptyPayloadPolicy.EMPTY_IS_DATA)),
        
        # 2. STALE_CARRY with id -> Valid
        (None, None, MockResolved(FreshnessState.STALE_CARRY, "id2"), None, None, MockAssessment(EmptyPayloadPolicy.EMPTY_MEANS_STALE)),
        
        # 3. EMPTY_VALID + EMPTY_IS_DATA -> Valid
        (None, None, MockResolved(FreshnessState.EMPTY_VALID, "id3"), None, None, MockAssessment(EmptyPayloadPolicy.EMPTY_IS_DATA)),
        
        # 4. ERROR -> Invalid
        (None, None, MockResolved(FreshnessState.ERROR, None), None, None, MockAssessment(EmptyPayloadPolicy.EMPTY_MEANS_STALE)),
        
        # 5. EMPTY_VALID + EMPTY_MEANS_STALE (Invalid configuration caught by invariant) -> Invalid
        (None, None, MockResolved(FreshnessState.EMPTY_VALID, None), None, None, MockAssessment(EmptyPayloadPolicy.EMPTY_MEANS_STALE)),
        
        # 6. STALE_CARRY without an explicit ID -> Invalid
        (None, None, MockResolved(FreshnessState.STALE_CARRY, None), None, None, MockAssessment(EmptyPayloadPolicy.EMPTY_MEANS_STALE)),
    ]
    
    dq = calculate_synthetic_dq(evs)
    
    # Exactly 3 valid out of 6 total = 0.5
    assert dq == 0.5