import pytest
from src.ingest_engine import _validate_config

@pytest.fixture
def valid_cfg():
    return {
        "ingestion": {"watchlist": ["AAPL"], "cadence_minutes": 5},
        "storage": {"duckdb_path": "", "cycle_lock_path": "", "writer_lock_path": ""},
        "system": {},
        "network": {},
        "model": {
            "weights": {
                "spot": 1.0,
                "oi_pressure": 0.5
            }
        },
        "validation": {
            "invalid_after_minutes": 60,
            "fallback_max_age_minutes": 15,
            "horizons_minutes": [15, 60],
            "alignment_tolerance_sec": 900,
            "emit_to_close_horizon": True,
            "use_default_required_features": False,
            "horizon_weights_source": "model",
            "horizon_weights": {},
            "horizon_weights_overrides": {
                "15": {},
                "60": {},
                "to_close": {}
            },
            "horizon_critical_features": {
                "15": ["spot"],
                "60": ["spot"],
                "to_close": ["spot"]
            }
        }
    }

def test_missing_required_key_fails(valid_cfg):
    del valid_cfg["validation"]["alignment_tolerance_sec"]
    with pytest.raises(KeyError, match="Missing validation.alignment_tolerance_sec"):
        _validate_config(valid_cfg)

def test_missing_to_close_criticals_fails_when_emitted(valid_cfg):
    del valid_cfg["validation"]["horizon_critical_features"]["to_close"]
    with pytest.raises(KeyError, match="Missing validation.horizon_critical_features for horizon 'to_close'"):
        _validate_config(valid_cfg)

def test_unknown_weight_key_fails_fast(valid_cfg):
    valid_cfg["model"]["weights"]["strike_flow_imbalance"] = 1.0
    with pytest.raises(ValueError, match="Unknown feature keys in config.*strike_flow_imbalance"):
        _validate_config(valid_cfg)

def test_unknown_critical_key_fails_fast(valid_cfg):
    valid_cfg["validation"]["horizon_critical_features"]["15"].append("some_unknown_feature")
    with pytest.raises(ValueError, match="Unknown feature keys in config.*some_unknown_feature"):
        _validate_config(valid_cfg)