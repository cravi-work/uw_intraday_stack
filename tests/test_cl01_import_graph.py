import sys
import pytest
import asyncio

def test_clean_module_imports():
    """Asserts that features and analytics load cleanly without CircularImport errors."""
    # Remove from sys.modules if already loaded to simulate cold start
    for mod in ["src.features", "src.analytics", "src.ingest_engine"]:
        if mod in sys.modules:
            del sys.modules[mod]
            
    import src.features
    import src.analytics
    import src.ingest_engine
    
    assert hasattr(src.features, "extract_all"), "extract_all missing from features"
    assert hasattr(src.analytics, "build_gex_levels"), "build_gex_levels missing from analytics"

def test_extraction_dry_run_no_import_errors():
    """Validates that extract_all runs end-to-end on minimal payload."""
    from src.endpoint_truth import EndpointContext
    from src.features import extract_all

    payloads = {
        1: [{"strike": 100, "gamma_exposure": 1000}, {"strike": 105, "gamma_exposure": -500}]
    }
    contexts = {
        1: EndpointContext(
            endpoint_id=1,
            method="GET",
            path="/api/stock/{ticker}/spot-exposures",
            operation_id="get_gex",
            signature="GET /api/stock/{ticker}/spot-exposures",
            used_event_id="test-uuid",
            payload_class="SUCCESS_HAS_DATA",
            freshness_state="FRESH",
            stale_age_min=0,
            na_reason=None
        )
    }
    
    f_rows, l_rows = extract_all(payloads, contexts)
    assert len(f_rows) > 0, "Failed to extract features"
    assert len(l_rows) > 0, "Failed to build GEX levels"