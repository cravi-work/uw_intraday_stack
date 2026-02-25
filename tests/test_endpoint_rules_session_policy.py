import pytest
import logging
from src.endpoint_rules import get_empty_policy, EmptyPayloadPolicy

def test_canonical_session_labels():
    """
    EVIDENCE: Asserts get_empty_policy() returns expected policies for representative 
    endpoints across PREMARKET, RTH, AFTERHOURS, and CLOSED.
    """
    # Test P_ALWAYS_DATA endpoint
    assert get_empty_policy("GET", "/api/darkpool/{ticker}", "PREMARKET") == EmptyPayloadPolicy.EMPTY_IS_DATA
    assert get_empty_policy("GET", "/api/darkpool/{ticker}", "RTH") == EmptyPayloadPolicy.EMPTY_IS_DATA
    
    # Test P_FLOW endpoint
    assert get_empty_policy("GET", "/api/stock/{ticker}/flow-recent", "RTH") == EmptyPayloadPolicy.EMPTY_IS_DATA
    assert get_empty_policy("GET", "/api/stock/{ticker}/flow-recent", "PREMARKET") == EmptyPayloadPolicy.EMPTY_MEANS_STALE
    assert get_empty_policy("GET", "/api/stock/{ticker}/flow-recent", "CLOSED") == EmptyPayloadPolicy.EMPTY_MEANS_STALE
    
    # Test P_STRUCTURAL endpoint
    assert get_empty_policy("GET", "/api/stock/{ticker}/max-pain", "RTH") == EmptyPayloadPolicy.EMPTY_INVALID
    assert get_empty_policy("GET", "/api/stock/{ticker}/max-pain", "AFTERHOURS") == EmptyPayloadPolicy.EMPTY_MEANS_STALE

def test_legacy_session_labels_explicit_failure(caplog):
    """
    EVIDENCE: Proves that an invalid session label (including legacy tags) triggers explicit failure 
    behavior and logs the contract violation rather than defaulting silently.
    """
    with caplog.at_level(logging.ERROR):
        with pytest.raises(ValueError, match="Unknown session label: REG. Allowed: PREMARKET, RTH, AFTERHOURS, CLOSED"):
            get_empty_policy("GET", "/api/stock/{ticker}/max-pain", "REG")
            
    assert "Session contract violation: Unknown session label 'REG'" in caplog.text

def test_unregistered_endpoint_fallback():
    """
    EVIDENCE: Unregistered paths correctly bypass session checking and return EMPTY_INVALID.
    """
    assert get_empty_policy("GET", "/api/stock/{ticker}/not-real-endpoint", "RTH") == EmptyPayloadPolicy.EMPTY_INVALID