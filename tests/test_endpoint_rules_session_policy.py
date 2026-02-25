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

def test_legacy_session_mapping():
    """
    EVIDENCE: Legacy session map successfully bridges old keys to Canonical versions.
    """
    assert get_empty_policy("GET", "/api/stock/{ticker}/max-pain", "REG") == EmptyPayloadPolicy.EMPTY_INVALID
    assert get_empty_policy("GET", "/api/stock/{ticker}/flow-recent", "PRE") == EmptyPayloadPolicy.EMPTY_MEANS_STALE

def test_unknown_session_label_explicit_failure(caplog):
    """
    EVIDENCE: Proves that an invalid session label triggers explicit failure 
    behavior and logs the contract violation rather than defaulting silently.
    """
    with caplog.at_level(logging.ERROR):
        with pytest.raises(ValueError, match="Unknown session label: INVALID_SESSION"):
            get_empty_policy("GET", "/api/stock/{ticker}/max-pain", "INVALID_SESSION")
            
    assert "Session contract violation: Unknown session label 'INVALID_SESSION'" in caplog.text

def test_unregistered_endpoint_fallback():
    """
    EVIDENCE: Unregistered paths correctly bypass session checking and return EMPTY_INVALID.
    """
    assert get_empty_policy("GET", "/api/stock/{ticker}/not-real-endpoint", "RTH") == EmptyPayloadPolicy.EMPTY_INVALID