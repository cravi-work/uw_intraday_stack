from unittest.mock import patch
from src.ingest_engine import IngestionEngine

@patch("src.ingest_engine.load_api_catalog")
def test_engine_initialization_contract(mock_load_catalog):
    # We mock load_api_catalog so we don't need real network/files to test init
    
    cfg = {
        "ingestion": {"watchlist": ["AAPL"], "cadence_minutes": 5},
        "storage": {"duckdb_path": ":memory:", "cycle_lock_path": "lock", "writer_lock_path": "wlock"},
        "system": {},
        "network": {},
        "validation": {
            "invalid_after_minutes": 60,
            "tolerance_minutes": 10,
            "max_horizon_drift_minutes": 10,
            "flat_threshold_pct": 0.001,
            "fallback_max_age_minutes": 15,"horizons_minutes": [15, 60]}
    }
    
    eng = IngestionEngine(cfg=cfg, catalog_path="api_catalog.generated.yaml")
    
    # Asserts the core lifecycle method is present on the engine
    assert hasattr(eng, "run_cycle")