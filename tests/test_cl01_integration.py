# tests/test_cl01_integration.py
import pytest
import datetime as dt
from unittest.mock import MagicMock, patch

def test_ingestion_dry_run_reaches_feature_insertion():
    from src.ingest_engine import IngestionEngine

    cfg = {
        "ingestion": {"watchlist": ["AAPL"], "cadence_minutes": 5, "enable_market_context": False},
        "storage": {"duckdb_path": ":memory:", "cycle_lock_path": "mock.lock", "writer_lock_path": "mock.lock"},
        "system": {},
        "network": {},
        "validation": {
            "alignment_tolerance_sec": 900,
            "use_default_required_features": False,
            "emit_to_close_horizon": False,
            "horizon_weights_source": "explicit",
            "horizons_minutes": [5],
            "horizon_critical_features": {"5": []},
            "horizon_weights": {"5": {"spot": 1.0}}
        }
    }

    with patch("src.ingest_engine.get_market_hours") as mock_gmh, \
         patch("src.ingest_engine.fetch_all") as mock_fetch, \
         patch("src.ingest_engine.load_endpoint_plan") as mock_lep, \
         patch("src.ingest_engine.load_api_catalog") as mock_lac, \
         patch("src.ingest_engine.validate_plan_coverage") as mock_vpc, \
         patch("src.ingest_engine.DbWriter") as mock_dbw_cls, \
         patch("src.ingest_engine.FileLock") as mock_fl:

        mock_mh = MagicMock()
        mock_mh.is_trading_day = True
        mock_mh.ingest_start_et = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=1)
        mock_mh.ingest_end_et = dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=1)
        mock_mh.get_session_label.return_value = "RTH"
        mock_mh.seconds_to_close.return_value = 3600
        mock_gmh.return_value = mock_mh

        mock_lep.return_value = {"plans": {"default": [{"name": "gex", "method": "GET", "path": "/api/stock/{ticker}/spot-exposures"}]}}

        mock_res = MagicMock()
        mock_res.requested_at_utc = dt.datetime.now(dt.timezone.utc).timestamp()
        mock_res.received_at_utc = mock_res.requested_at_utc + 0.1
        mock_res.status_code = 200
        mock_res.payload_hash = "mock_hash"
        mock_res.payload_json = [{"strike": 100, "gamma_exposure": 5000}]
        mock_res.retry_count = 0
        mock_res.error_type = None
        mock_res.error_message = None

        mock_call = MagicMock()
        mock_call.method = "GET"
        mock_call.path = "/api/stock/{ticker}/spot-exposures"

        async def fake_fetch(*args, **kwargs):
            return [("AAPL", mock_call, "sig", {}, mock_res, None)]
        mock_fetch.side_effect = fake_fetch

        mock_db = MagicMock()
        mock_dbw_cls.return_value = mock_db
        mock_db.writer.return_value.__enter__.return_value = MagicMock()
        mock_db.insert_raw_event.return_value = "event-uuid"
        mock_db.get_payloads_by_event_ids.return_value = {"event-uuid": mock_res.payload_json}

        engine = IngestionEngine(cfg=cfg, catalog_path="dummy.yaml", config_path="dummy.yaml")
        engine.run_cycle()

        mock_db.insert_features.assert_called()