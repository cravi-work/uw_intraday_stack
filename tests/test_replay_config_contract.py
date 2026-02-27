# tests/test_replay_config_contract.py
import pytest
from src.replay_engine import run_replay

def test_replay_fails_on_invalid_config():
    """
    Task 5 Proof: If run_replay is handed an invalid config, it must raise the 
    underlying _validate_config validation error explicitly and immediately, 
    without "catching" it and without attempting to connect to DuckDB.
    """
    invalid_cfg = {
        "ingestion": {"watchlist": ["AAPL"], "cadence_minutes": 5},
        "storage": {"duckdb_path": "dummy.duckdb", "cycle_lock_path": "mock.lock", "writer_lock_path": "mock.lock"},
        "system": {},
        "network": {},
        "validation": {
            # DELIBERATELY MISSING: "alignment_tolerance_sec"
            "use_default_required_features": False,
            "emit_to_close_horizon": False,
            "horizons_minutes": [15],
            "horizon_weights_source": "explicit",
            "horizon_critical_features": {"15": []},
            "horizon_weights": {"15": {"spot": 1.0}},
            "fallback_max_age_minutes": 15,
            "invalid_after_minutes": 60
        }
    }

    with pytest.raises(KeyError, match="Missing validation.alignment_tolerance_sec"):
        # Must fail before opening duckdb
        run_replay("dummy_path.duckdb", "AAPL", cfg=invalid_cfg)

def test_replay_fails_on_missing_hidden_defaults():
    """
    Task 6 Proof: The system must hard-fail if the previously hidden defaults 
    ('invalid_after_minutes' or 'fallback_max_age_minutes') are missing from the explicit validation config block.
    """
    invalid_cfg = {
        "ingestion": {"watchlist": ["AAPL"], "cadence_minutes": 5},
        "storage": {"duckdb_path": "dummy.duckdb", "cycle_lock_path": "mock.lock", "writer_lock_path": "mock.lock"},
        "system": {},
        "network": {},
        "validation": {
            "alignment_tolerance_sec": 900,
            "use_default_required_features": False,
            "emit_to_close_horizon": False,
            "horizons_minutes": [15],
            "horizon_weights_source": "explicit",
            "horizon_critical_features": {"15": []},
            "horizon_weights": {"15": {"spot": 1.0}},
            # DELIBERATELY MISSING: "fallback_max_age_minutes"
            "invalid_after_minutes": 60
        }
    }

    with pytest.raises(KeyError, match="Missing validation.fallback_max_age_minutes"):
        run_replay("dummy_path.duckdb", "AAPL", cfg=invalid_cfg)
        
    invalid_cfg["validation"]["fallback_max_age_minutes"] = 15
    del invalid_cfg["validation"]["invalid_after_minutes"]
    
    with pytest.raises(KeyError, match="Missing validation.invalid_after_minutes"):
        run_replay("dummy_path.duckdb", "AAPL", cfg=invalid_cfg)
        
    invalid_cfg["validation"]["invalid_after_minutes"] = "a_string"
    
    with pytest.raises(ValueError, match="validation.invalid_after_minutes must be a positive integer"):
        run_replay("dummy_path.duckdb", "AAPL", cfg=invalid_cfg)