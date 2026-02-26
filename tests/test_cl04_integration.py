# tests/test_cl04_integration.py
import pytest
import datetime as dt
import logging
from unittest.mock import MagicMock, patch

def test_expanded_feature_persistence_with_lineage(caplog):
    from src.ingest_engine import IngestionEngine

    cfg = {
        "ingestion": {"watchlist": ["AAPL"], "cadence_minutes": 5, "enable_market_context": False},
        "storage": {"duckdb_path": ":memory:", "cycle_lock_path": "mock.lock", "writer_lock_path": "mock.lock"},
        "system": {},
        "network": {},
        "validation": {
            "horizons_minutes": [5],
            "use_default_required_features": False,
            "emit_to_close_horizon": False,
            "horizon_critical_features": {}
        }
    }

    with patch("src.ingest_engine.get_market_hours") as mock_gmh, \
         patch("src.ingest_engine.fetch_all") as mock_fetch, \
         patch("src.ingest_engine.load_endpoint_plan") as mock_lep, \
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

        mock_lep.return_value = {"plans": {"default": [
            {"name": "dp", "method": "GET", "path": "/api/darkpool/{ticker}"},
            {"name": "oi", "method": "GET", "path": "/api/stock/{ticker}/oi-per-strike"},
            {"name": "ts", "method": "GET", "path": "/api/stock/{ticker}/volatility/term-structure"},
            {"name": "skew", "method": "GET", "path": "/api/stock/{ticker}/historical-risk-reversal-skew"}
        ]}}

        mock_call_dp = MagicMock()
        mock_call_dp.method, mock_call_dp.path = "GET", "/api/darkpool/{ticker}"
        
        mock_res_dp = MagicMock()
        mock_res_dp.requested_at_utc = dt.datetime.now(dt.timezone.utc).timestamp()
        mock_res_dp.received_at_utc = mock_res_dp.requested_at_utc + 0.1
        mock_res_dp.status_code = 200
        mock_res_dp.payload_hash = "dp_hash"
        mock_res_dp.payload_json = [{"price": 100, "volume": 500}] 
        mock_res_dp.retry_count = 0
        mock_res_dp.error_type = None
        mock_res_dp.error_message = None

        async def fake_fetch(*args, **kwargs):
            return [("AAPL", mock_call_dp, "sig1", {}, mock_res_dp, None)]
        mock_fetch.side_effect = fake_fetch

        mock_db = MagicMock()
        mock_dbw_cls.return_value = mock_db
        mock_db.writer.return_value.__enter__.return_value = MagicMock()
        mock_db.get_payloads_by_event_ids.return_value = {"uuid1": mock_res_dp.payload_json}

        engine = IngestionEngine(cfg=cfg, catalog_path="api_catalog.generated.yaml", config_path="dummy.yaml")
        
        with patch('src.ingest_engine.extract_all') as mock_extract:
            
            valid_meta = {
                "source_endpoints": [], "freshness_state": "FRESH", "stale_age_min": 0, "na_reason": None, "details": {},
                "metric_lineage": {"effective_ts_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
            }
            
            mock_extract.return_value = (
                [
                    {"feature_key": "darkpool_pressure", "feature_value": 50000.0, "meta_json": valid_meta},
                    {"feature_key": "vol_term_slope", "feature_value": 0.10, "meta_json": valid_meta},
                    {"feature_key": "vol_skew", "feature_value": 1.5, "meta_json": valid_meta}
                ],
                [{"level_type": "OI_MAX_WALL", "price": 100, "magnitude": 1000, "meta_json": valid_meta}]
            )
            
            with caplog.at_level(logging.INFO):
                engine.run_cycle()
            
            mock_db.insert_features.assert_called()
            mock_db.insert_levels.assert_called()
            
            inserted_features = mock_db.insert_features.call_args[0][2]
            keys = [f["feature_key"] for f in inserted_features]
            assert "vol_term_slope" in keys
            assert "vol_skew" in keys
            assert "darkpool_pressure" in keys