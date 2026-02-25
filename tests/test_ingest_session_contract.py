import pytest
import datetime as dt
import logging
from unittest.mock import MagicMock, patch
from src.ingest_engine import IngestionEngine

def test_valid_session_labels_pass(caplog):
    """
    EVIDENCE: Valid canonical labels allow the cycle to proceed (no fast-fail).
    """
    cfg = {
        "ingestion": {"watchlist": ["AAPL"], "cadence_minutes": 5, "enable_market_context": False},
        "storage": {"duckdb_path": ":memory:", "cycle_lock_path": "mock.lock", "writer_lock_path": "mock.lock"},
        "system": {},
        "network": {},
        "validation": {"horizons_minutes": [5]}
    }

    with patch("src.ingest_engine.get_market_hours") as mock_gmh, \
         patch("src.ingest_engine.fetch_all") as mock_fetch, \
         patch("src.ingest_engine.load_endpoint_plan"), \
         patch("src.ingest_engine.validate_plan_coverage"), \
         patch("src.ingest_engine.load_api_catalog"), \
         patch("src.ingest_engine.DbWriter"), \
         patch("src.ingest_engine.FileLock"):
        
        mock_mh = MagicMock()
        mock_mh.is_trading_day = True
        mock_mh.ingest_start_et = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=1)
        mock_mh.ingest_end_et = dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=1)
        mock_mh.get_session_label.return_value = "RTH"
        mock_gmh.return_value = mock_mh

        engine = IngestionEngine(cfg=cfg, catalog_path="dummy.yaml", config_path="dummy.yaml")
        
        with caplog.at_level(logging.ERROR):
            engine.run_cycle()
        
        assert "Session contract violation" not in caplog.text

def test_invalid_session_label_fails_fast(caplog):
    """
    EVIDENCE: Invalid session label triggers contract violation counter,
    logs explicit context, and fails cycle before any database/prediction writes.
    """
    cfg = {
        "ingestion": {"watchlist": ["AAPL"], "cadence_minutes": 5, "enable_market_context": False},
        "storage": {"duckdb_path": ":memory:", "cycle_lock_path": "mock.lock", "writer_lock_path": "mock.lock"},
        "system": {"mode": "test_replay"},
        "network": {},
        "validation": {"horizons_minutes": [5]}
    }

    with patch("src.ingest_engine.get_market_hours") as mock_gmh, \
         patch("src.ingest_engine.fetch_all") as mock_fetch, \
         patch("src.ingest_engine.load_endpoint_plan"), \
         patch("src.ingest_engine.validate_plan_coverage"), \
         patch("src.ingest_engine.load_api_catalog"), \
         patch("src.ingest_engine.DbWriter") as mock_dbw_cls, \
         patch("src.ingest_engine.FileLock"):
        
        mock_mh = MagicMock()
        mock_mh.is_trading_day = True
        mock_mh.ingest_start_et = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=1)
        mock_mh.ingest_end_et = dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=1)
        mock_mh.get_session_label.return_value = "REG" # Invalid legacy label
        mock_gmh.return_value = mock_mh

        mock_db = MagicMock()
        mock_dbw_cls.return_value = mock_db

        engine = IngestionEngine(cfg=cfg, catalog_path="dummy.yaml", config_path="dummy.yaml")
        
        with caplog.at_level(logging.ERROR):
            engine.run_cycle()
        
        assert "Session contract violation" in caplog.text
        
        violation_record = next(r for r in caplog.records if "Session contract violation" in r.message)
        assert getattr(violation_record, "counter") == "session_contract_violation_count"
        assert getattr(violation_record, "raw_session_label") == "REG"
        assert getattr(violation_record, "processing_mode") == "test_replay"
        assert "AAPL" in getattr(violation_record, "tickers")

        # Verify hard fail (fetch and db write never called)
        mock_fetch.assert_not_called()
        mock_db.insert_prediction.assert_not_called()