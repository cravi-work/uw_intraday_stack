import pytest
import datetime as dt
import logging
from unittest.mock import MagicMock, patch
from src.ingest_engine import IngestionEngine

@pytest.fixture
def mock_engine_env():
    cfg = {
        "ingestion": {"watchlist": ["AAPL"], "cadence_minutes": 5, "enable_market_context": False},
        "storage": {"duckdb_path": ":memory:", "cycle_lock_path": "mock.lock", "writer_lock_path": "mock.lock"},
        "system": {},
        "network": {},
        "validation": {
            "horizons_minutes": [5],
            "horizon_critical_features": {"5": ["spot"]},
            "horizon_weights": {"5": {"spot": 1.0, "oi_pressure": 1.0}}
        }
    }
    return cfg

def _run_with_features(cfg, features, caplog, mock_fetch_len=10, mock_fetch_success=10):
    with patch("src.ingest_engine.get_market_hours") as mock_gmh, \
         patch("src.ingest_engine.fetch_all") as mock_fetch, \
         patch("src.ingest_engine.load_endpoint_plan") as mock_lep, \
         patch("src.ingest_engine.load_api_catalog"), \
         patch("src.ingest_engine.validate_plan_coverage"), \
         patch("src.ingest_engine.DbWriter") as mock_dbw_cls, \
         patch("src.ingest_engine.FileLock"):

        mock_mh = MagicMock()
        mock_mh.is_trading_day = True
        mock_mh.ingest_start_et = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=1)
        mock_mh.ingest_end_et = dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=1)
        mock_mh.get_session_label.return_value = "RTH"
        mock_mh.seconds_to_close.return_value = 3600
        mock_gmh.return_value = mock_mh

        mock_lep.return_value = {"plans": {"default": []}}
        
        # Simulate fetch events to drive endpoint_coverage
        events = []
        for i in range(mock_fetch_len):
            m = MagicMock()
            m.requested_at_utc = dt.datetime.now(dt.timezone.utc).timestamp()
            m.received_at_utc = m.requested_at_utc + 0.1
            m.status_code = 200 if i < mock_fetch_success else 500
            m.payload_hash = "mock_hash"
            m.payload_json = []
            m.retry_count = 0
            m.error_type = None
            m.error_message = None
            
            c = MagicMock()
            c.method = "GET"
            c.path = f"/api/mock/{i}"
            events.append(("AAPL", c, f"sig{i}", {}, m, None))

        async def fake_fetch(*args, **kwargs): return events
        mock_fetch.side_effect = fake_fetch

        mock_db = MagicMock()
        mock_dbw_cls.return_value = mock_db
        mock_db.writer.return_value.__enter__.return_value = MagicMock()
        mock_db.get_payloads_by_event_ids.return_value = {}

        engine = IngestionEngine(cfg=cfg, catalog_path="dummy.yaml", config_path="dummy.yaml")
        
        with patch('src.ingest_engine.extract_all') as mock_extract:
            mock_extract.return_value = (features, [])
            with caplog.at_level(logging.INFO):
                engine.run_cycle()
                
        return mock_db

def test_dq_is_low_when_critical_missing_despite_high_endpoint_coverage(mock_engine_env, caplog):
    """
    Scenario A: many endpoints valid but critical weighted feature missing -> decision DQ is low.
    """
    # 10 out of 10 endpoints succeed (100% fetch coverage)
    # But `spot` (required for 5m horizon) is missing from extracted features
    valid_meta = {"source_endpoints": [], "freshness_state": "FRESH", "stale_age_min": 0, "na_reason": None, "details": {}}
    now_utc = dt.datetime.now(dt.timezone.utc).isoformat()
    
    features = [
        {"feature_key": "oi_pressure", "feature_value": 1000.0, "meta_json": {**valid_meta, "metric_lineage": {"effective_ts_utc": now_utc}}}
    ]
    
    mock_db = _run_with_features(mock_engine_env, features, caplog, mock_fetch_len=10, mock_fetch_success=10)
    
    # Verify endpoint_coverage was logged properly as 100%
    assert "ratio': 1.0" in caplog.text 
    
    # Verify decision_dq is low (0.5 because 1 out of 2 target features was found)
    assert "dq_reasons_horizon_5" in caplog.text
    assert "spot_missing_or_invalid" in caplog.text
    
    pred_call = mock_db.insert_prediction.call_args[0][1]
    assert pred_call["meta_json"]["dq_eff"] == 0.5 

def test_dq_acceptable_when_endpoints_fail_but_required_features_present(mock_engine_env, caplog):
    """
    Scenario B: few endpoints overall, but all weighted/critical features valid and aligned -> DQ is acceptable.
    """
    # 10 endpoints fetched, but 8 failed (only 20% endpoint coverage)
    # However, the 2 that succeeded yielded perfectly clean `spot` and `oi_pressure`
    valid_meta = {"source_endpoints": [], "freshness_state": "FRESH", "stale_age_min": 0, "na_reason": None, "details": {}}
    now_utc = dt.datetime.now(dt.timezone.utc).isoformat()
    
    features = [
        {"feature_key": "spot", "feature_value": 150.0, "meta_json": {**valid_meta, "metric_lineage": {"effective_ts_utc": now_utc}}},
        {"feature_key": "oi_pressure", "feature_value": 1000.0, "meta_json": {**valid_meta, "metric_lineage": {"effective_ts_utc": now_utc}}}
    ]
    
    mock_db = _run_with_features(mock_engine_env, features, caplog, mock_fetch_len=10, mock_fetch_success=2)
    
    # Endpoint coverage logged low
    assert "ratio': 0.2" in caplog.text 
    
    # Decision DQ is perfect (1.0) because both target features perfectly resolved
    pred_call = mock_db.insert_prediction.call_args[0][1]
    assert pred_call["meta_json"]["dq_eff"] == 1.0
    assert pred_call["decision_state"] != "NO_SIGNAL"

def test_stale_features_reduce_dq(mock_engine_env, caplog):
    """
    Scenario C: stale/misaligned weighted features reduce DQ even if fetched successfully.
    """
    valid_meta = {"source_endpoints": [], "freshness_state": "ERROR", "stale_age_min": 0, "na_reason": None, "details": {}}
    now_utc = dt.datetime.now(dt.timezone.utc).isoformat()
    
    features = [
        {"feature_key": "spot", "feature_value": 150.0, "meta_json": {**valid_meta, "metric_lineage": {"effective_ts_utc": now_utc}}}, # ERROR freshness
        {"feature_key": "oi_pressure", "feature_value": 1000.0, "meta_json": {**valid_meta, "metric_lineage": {"effective_ts_utc": now_utc}}} # ERROR freshness
    ]
    
    mock_db = _run_with_features(mock_engine_env, features, caplog)
    
    assert "dq_reasons_horizon_5" in caplog.text
    assert "spot_bad_freshness_ERROR" in caplog.text
    
    pred_call = mock_db.insert_prediction.call_args[0][1]
    assert pred_call["meta_json"]["dq_eff"] == 0.0 # Both features rejected for freshness