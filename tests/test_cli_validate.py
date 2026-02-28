# tests/test_cli_validate.py
import sys
import pytest
from unittest.mock import patch, MagicMock
from src.main import main

def test_validate_cli_invalid_config_exits(capsys):
    test_args = ["main.py", "validate", "--config", "dummy.yaml"]
    with patch("sys.argv", test_args), \
         patch("src.main.load_yaml") as mock_load:
         
         mock_load.return_value.raw = {"storage": {"duckdb_path": "test.duckdb"}}
         
         with pytest.raises(SystemExit) as exc:
             main()
             
         assert exc.value.code == 1
         stderr = capsys.readouterr().err
         assert "Config Validation Error:" in stderr

def test_validate_cli_valid_config_routes():
    test_args = ["main.py", "validate", "--config", "dummy.yaml"]
    
    with patch("sys.argv", test_args), \
         patch("src.main.load_yaml") as mock_load, \
         patch("src.main.duckdb.connect"), \
         patch("src.validator.validate_pending") as mock_vp:  # FIX: Patched true module location
         
         valid_cfg = {
             "ingestion": {"watchlist": ["AAPL"], "cadence_minutes": 5},
             "storage": {"duckdb_path": "my_val.duckdb", "cycle_lock_path": "x", "writer_lock_path": "y"},
             "system": {},
             "network": {},
             "validation": {
                 "horizons_minutes": [15],
                 "alignment_tolerance_sec": 900,
                 "emit_to_close_horizon": False,
                 "use_default_required_features": False,
                 "invalid_after_minutes": 60,
                 "fallback_max_age_minutes": 15,
                 "tolerance_minutes": 10,
                 "max_horizon_drift_minutes": 15,
                 "flat_threshold_pct": 0.001,
                 "horizon_weights_source": "explicit",
                 "horizon_critical_features": {"15": []},
                 "horizon_weights": {"15": {"spot": 1.0}}
             }
         }
         mock_load.return_value.raw = valid_cfg
         
         mock_res = MagicMock()
         mock_res.updated = 5
         mock_res.skipped = 2
         mock_vp.return_value = mock_res
         
         with pytest.raises(SystemExit) as exc:
             main()
             
         assert exc.value.code == 0
         mock_vp.assert_called_once()
         kwargs = mock_vp.call_args.kwargs
         assert kwargs["flat_threshold_pct"] == 0.001
         assert kwargs["tolerance_minutes"] == 10
         assert kwargs["max_horizon_drift_minutes"] == 15