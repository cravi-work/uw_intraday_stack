import sys
import pytest
from unittest.mock import MagicMock

def test_clean_module_imports():
    """
    EVIDENCE: Asserts that features, analytics, and ingest_engine load cleanly
    without CircularImport errors. Validates the unidirectional contract.
    """
    
    # Remove from sys.modules if already loaded to simulate cold start
    modules_to_purge = ["src.features", "src.analytics", "src.ingest_engine", "src.na"]
    for mod in modules_to_purge:
        if mod in sys.modules:
            del sys.modules[mod]
            
    import src.features
    import src.analytics
    
    # Dynamic import to test engine's stability with new interface
    import src.ingest_engine
    
    assert hasattr(src.features, "extract_all"), "extract_all missing from features"
    assert hasattr(src.analytics, "build_gex_levels"), "build_gex_levels missing from analytics"

def test_extraction_dry_run_no_import_errors():
    """
    EVIDENCE: Validates that extract_all runs end-to-end on minimal payload
    and executes analytics function cleanly at runtime.
    """
    from src.features import extract_all
    
    # Mocking EndpointContext to avoid importing it strictly for testing payload routing
    mock_ctx = MagicMock()
    mock_ctx.path = "/api/stock/{ticker}/spot-exposures"
    mock_ctx.freshness_state = "FRESH"
    mock_ctx.stale_age_min = 0
    mock_ctx.method = "GET"
    mock_ctx.operation_id = "test"
    mock_ctx.signature = "test"
    mock_ctx.used_event_id = "uuid-123"
    mock_ctx.na_reason = None

    payloads = {
        1: [{"strike": 100, "gamma_exposure": 1000}, {"strike": 105, "gamma_exposure": -500}]
    }
    contexts = {1: mock_ctx}
    
    f_rows, l_rows = extract_all(payloads, contexts)
    
    assert len(f_rows) > 0, "Failed to extract core features from GEX path"
    assert len(l_rows) > 0, "Failed to build GEX levels via analytics.py call"