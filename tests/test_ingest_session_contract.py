# tests/test_ingest_session_contract.py
import pytest
import datetime as dt
import logging
from unittest.mock import MagicMock, patch

import src.ingest_engine as ie_mod
from src.ingest_engine import IngestionEngine

def test_valid_session_labels_pass(caplog):
    cfg = {
        "ingestion": {
            "watchlist": ["AAPL"], "cadence_minutes": 5, "enable_market_context": False,
            "premarket_start_et": "04:00", "regular_start_et": "09:30", "regular_end_et": "16:00",
            "afterhours_end_et": "20:00", "ingest_start_et": "04:00", "ingest_end_et": "20:00"
        },
        "storage": {"duckdb_path": ":memory:", "cycle_lock_path": "mock.lock", "writer_lock_path": "mock.lock"},
        "system": {},
        "network": {},
        "validation": {
            "invalid_after_minutes": 60,
            "fallback_max_age_minutes": 15,
            "alignment_tolerance_sec": 900,
            "use_default_required_features": False,
            "emit_to_close_horizon": False,
            "horizon_weights_source": "explicit",
            "horizons_minutes": [5],
            "horizon_critical_features": {"5": []},
            "horizon_weights": {"5": {"spot": 1.0}}
        }
    }

    with patch.object(ie_mod, "get_market_hours") as mock_gmh, \
         patch.object(ie_mod, "fetch_all") as mock_fetch, \
         patch.object(ie_mod, "load_endpoint_plan"), \
         patch.object(ie_mod, "validate_plan_coverage"), \
         patch.object(ie_mod, "load_api_catalog"), \
         patch.object(ie_mod, "DbWriter"), \
         patch.object(ie_mod, "FileLock"):
        
        mock_mh = MagicMock()
        mock_mh.is_trading_day = True
        mock_mh.ingest_start_et = dt.datetime(2000, 1, 1, tzinfo=dt.timezone.utc)
        mock_mh.ingest_end_et = dt.datetime(2100, 1, 1, tzinfo=dt.timezone.utc)
        mock_mh.get_session_label.return_value = "RTH"
        mock_gmh.return_value = mock_mh

        async def fake_fetch(*args, **kwargs): return []
        mock_fetch.side_effect = fake_fetch

        engine = IngestionEngine(cfg=cfg, catalog_path="api_catalog.generated.yaml", config_path="src/config/config.yaml")
        
        with caplog.at_level(logging.ERROR):
            engine.run_cycle()
        
        assert "Session contract violation" not in caplog.text

def test_invalid_session_label_fails_fast(caplog):
    cfg = {
        "ingestion": {
            "watchlist": ["AAPL"], "cadence_minutes": 5, "enable_market_context": False,
            "premarket_start_et": "04:00", "regular_start_et": "09:30", "regular_end_et": "16:00",
            "afterhours_end_et": "20:00", "ingest_start_et": "04:00", "ingest_end_et": "20:00"
        },
        "storage": {"duckdb_path": ":memory:", "cycle_lock_path": "mock.lock", "writer_lock_path": "mock.lock"},
        "system": {"mode": "test_replay"},
        "network": {},
        "validation": {
            "invalid_after_minutes": 60,
            "fallback_max_age_minutes": 15,
            "alignment_tolerance_sec": 900,
            "use_default_required_features": False,
            "emit_to_close_horizon": False,
            "horizon_weights_source": "explicit",
            "horizons_minutes": [5],
            "horizon_critical_features": {"5": []},
            "horizon_weights": {"5": {"spot": 1.0}}
        }
    }

    with patch.object(ie_mod, "get_market_hours") as mock_gmh, \
         patch.object(ie_mod, "fetch_all") as mock_fetch, \
         patch.object(ie_mod, "load_endpoint_plan"), \
         patch.object(ie_mod, "validate_plan_coverage"), \
         patch.object(ie_mod, "load_api_catalog"), \
         patch.object(ie_mod, "DbWriter") as mock_dbw_cls, \
         patch.object(ie_mod, "FileLock"):
        
        mock_mh = MagicMock()
        mock_mh.is_trading_day = True
        mock_mh.ingest_start_et = dt.datetime(2000, 1, 1, tzinfo=dt.timezone.utc)
        mock_mh.ingest_end_et = dt.datetime(2100, 1, 1, tzinfo=dt.timezone.utc)
        mock_mh.get_session_label.return_value = "REG" # Invalid legacy label
        mock_gmh.return_value = mock_mh

        mock_db = MagicMock()
        mock_dbw_cls.return_value = mock_db

        engine = IngestionEngine(cfg=cfg, catalog_path="api_catalog.generated.yaml", config_path="src/config/config.yaml")
        
        with caplog.at_level(logging.ERROR):
            engine.run_cycle()
        
        assert "Session contract violation" in caplog.text
        
        violation_record = next(r for r in caplog.records if "Session contract violation" in r.message)
        assert getattr(violation_record, "counter") == "session_contract_violation_count"
        assert getattr(violation_record, "raw_session_label") == "REG"
        assert getattr(violation_record, "processing_mode") == "test_replay"
        assert "AAPL" in getattr(violation_record, "tickers")

        mock_fetch.assert_not_called()
        mock_db.insert_prediction.assert_not_called()