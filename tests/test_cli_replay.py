# tests/test_cli_replay.py
import sys
import pytest
from unittest.mock import patch
from src.main import main

def test_replay_cli_invalid_config_exits(capsys):
    """
    Task 9 Proof: Ensure replay hard-fails on invalid config, prints an explicit error,
    and returns exit code 1 (without attempting to execute).
    """
    test_args = ["main.py", "replay", "AAPL", "--config", "dummy.yaml"]
    
    with patch("sys.argv", test_args), \
         patch("src.main.load_yaml") as mock_load:
         
         mock_load.return_value.raw = {"storage": {"duckdb_path": "test.duckdb"}}
         
         with pytest.raises(SystemExit) as exc:
             main()
             
         assert exc.value.code == 1
         stderr = capsys.readouterr().err
         assert "Config Validation Error:" in stderr

def test_replay_cli_validation_intercept(capsys):
    """
    Ensure the CLI catches KeyError from _validate_config inside run_replay and translates it to exit code 1.
    """
    test_args = ["main.py", "replay", "AAPL", "--config", "dummy.yaml"]
    
    with patch("sys.argv", test_args), \
         patch("src.main.load_yaml") as mock_load, \
         patch("src.replay_engine.run_replay") as mock_run:
         
         mock_load.return_value.raw = {"storage": {"duckdb_path": "test.duckdb"}}
         mock_run.side_effect = KeyError("Missing validation.alignment_tolerance_sec")
         
         with pytest.raises(SystemExit) as exc:
             main()
             
         assert exc.value.code == 1
         stderr = capsys.readouterr().err
         assert "Config Validation Error:" in stderr
         assert "Missing validation.alignment_tolerance_sec" in stderr

def test_replay_cli_valid_config_routes():
    """
    Task 9 Proof: A valid config properly invokes run_replay with the db_path, ticker, and cfg.
    """
    test_args = ["main.py", "replay", "AAPL", "--config", "dummy.yaml"]
    
    with patch("sys.argv", test_args), \
         patch("src.main.load_yaml") as mock_load, \
         patch("src.replay_engine.run_replay") as mock_run:
         
         valid_cfg = {
             "ingestion": {"watchlist": ["AAPL"], "cadence_minutes": 5},
             "storage": {"duckdb_path": "my_replay.duckdb", "cycle_lock_path": "x", "writer_lock_path": "y"},
             "system": {},
             "network": {},
             "validation": {
                 "horizons_minutes": [15],
                 "alignment_tolerance_sec": 900,
                 "emit_to_close_horizon": False,
                 "use_default_required_features": False,
                 "invalid_after_minutes": 60,
            "tolerance_minutes": 10,
            "max_horizon_drift_minutes": 10,
            "flat_threshold_pct": 0.001,
                 "fallback_max_age_minutes": 15,
                 "horizon_weights_source": "explicit",
                 "horizon_critical_features": {"15": []},
                 "horizon_weights": {"15": {"spot": 1.0}}
             }
         }
         mock_load.return_value.raw = valid_cfg
         
         with pytest.raises(SystemExit) as exc:
             main()
             
         assert exc.value.code == 0
         mock_run.assert_called_once_with("my_replay.duckdb", "AAPL", cfg=valid_cfg)