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
            "use_default_required_features": False,
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
    valid_meta = {"source_endpoints": [], "freshness_state": "FRESH", "stale_age_min": 0, "na_reason": None, "details": {}}
    now_utc = dt.datetime.now(dt.timezone.utc).isoformat()
    
    features = [
        {"feature_key": "oi_pressure", "feature_value": 1000.0, "meta_json": {**valid_meta, "metric_lineage": {"effective_ts_utc": now_utc}}}
    ]
    
    mock_db = _run_with_features(mock_engine_env, features, caplog, mock_fetch_len=10, mock_fetch_success=10)
    
    assert "ratio': 1.0" in caplog.text 
    
    assert "dq_reasons_horizon_5" in caplog.text
    assert "spot_missing_or_invalid" in caplog.text
    
    pred_call = mock_db.insert_prediction.call_args[0][1]
    assert pred_call["meta_json"]["dq_eff"] == 0.5 
    
    # TASK 12: Verification points
    assert pred_call["decision_state"] == "NO_SIGNAL"
    assert pred_call["data_quality_state"] == "INVALID"
    assert "gate_state_transition_horizon_5: VALID -> INVALID" in caplog.text

def test_dq_acceptable_when_endpoints_fail_but_required_features_present(mock_engine_env, caplog):
    """
    Scenario B: few endpoints overall, but all weighted/critical features valid and aligned -> DQ is acceptable.
    """
    valid_meta = {"source_endpoints": [], "freshness_state": "FRESH", "stale_age_min": 0, "na_reason": None, "details": {}}
    now_utc = dt.datetime.now(dt.timezone.utc).isoformat()
    
    features = [
        {"feature_key": "spot", "feature_value": 150.0, "meta_json": {**valid_meta, "metric_lineage": {"effective_ts_utc": now_utc}}},
        {"feature_key": "oi_pressure", "feature_value": 1000.0, "meta_json": {**valid_meta, "metric_lineage": {"effective_ts_utc": now_utc}}}
    ]
    
    mock_db = _run_with_features(mock_engine_env, features, caplog, mock_fetch_len=10, mock_fetch_success=2)
    
    assert "ratio': 0.2" in caplog.text 
    assert "resolved_horizon_5_features" in caplog.text
    
    pred_call = mock_db.insert_prediction.call_args[0][1]
    assert pred_call["meta_json"]["dq_eff"] == 1.0
    
    # TASK 12: Verification points
    assert pred_call["decision_state"] != "NO_SIGNAL"
    assert pred_call["data_quality_state"] == "VALID"
    # Should NOT have a transition log because it stays VALID
    assert "gate_state_transition_horizon_5" not in caplog.text

def test_stale_features_reduce_dq(mock_engine_env, caplog):
    """
    Scenario C: STALE_CARRY inputs trigger partial penalty per feature based on age, reducing final DQ,
    and forcing the data_quality_state to 'PARTIAL'.
    """
    valid_meta = {"source_endpoints": [], "freshness_state": "STALE_CARRY", "stale_age_min": 30, "na_reason": None, "details": {}}
    now_utc = dt.datetime.now(dt.timezone.utc).isoformat()
    
    features = [
        {"feature_key": "spot", "feature_value": 150.0, "meta_json": {**valid_meta, "metric_lineage": {"effective_ts_utc": now_utc}}},
        {"feature_key": "oi_pressure", "feature_value": 1000.0, "meta_json": {**valid_meta, "metric_lineage": {"effective_ts_utc": now_utc}}}
    ]
    
    mock_db = _run_with_features(mock_engine_env, features, caplog)
    
    assert "dq_reasons_horizon_5" in caplog.text
    assert "spot_stale_carry_age_30m" in caplog.text
    
    pred_call = mock_db.insert_prediction.call_args[0][1]
    assert pred_call["meta_json"]["dq_eff"] < 1.0 
    
    # TASK 12: Verification points
    assert pred_call["data_quality_state"] == "PARTIAL" 
    assert "gate_state_transition_horizon_5: VALID -> PARTIAL" in caplog.text

def test_stale_non_critical_lowers_confidence(mock_engine_env, caplog):
    """
    EVIDENCE: Stale non-critical weighted feature lowers confidence without forcing NO_SIGNAL.
    """
    valid_meta_fresh = {"source_endpoints": [], "freshness_state": "FRESH", "stale_age_min": 0, "na_reason": None, "details": {}}
    valid_meta_stale = {"source_endpoints": [], "freshness_state": "STALE_CARRY", "stale_age_min": 15, "na_reason": None, "details": {}}
    now_utc = dt.datetime.now(dt.timezone.utc).isoformat()
    
    features = [
        {"feature_key": "spot", "feature_value": 150.0, "meta_json": {**valid_meta_fresh, "metric_lineage": {"effective_ts_utc": now_utc}}},
        {"feature_key": "oi_pressure", "feature_value": 1000.0, "meta_json": {**valid_meta_stale, "metric_lineage": {"effective_ts_utc": now_utc}}}
    ]
    
    mock_engine_env["validation"]["horizon_critical_features"] = {"5": ["spot"]}
    mock_engine_env["validation"]["horizon_weights"] = {"5": {"spot": 1.0, "oi_pressure": 1.0}}
    mock_engine_env["validation"]["use_default_required_features"] = False
    
    mock_db = _run_with_features(mock_engine_env, features, caplog)
    
    assert "dq_reasons_horizon_5" in caplog.text
    assert "oi_pressure_stale_carry_age_15m" in caplog.text
    
    pred_call = mock_db.insert_prediction.call_args[0][1]
    assert pred_call["decision_state"] != "NO_SIGNAL"
    assert pred_call["meta_json"]["dq_eff"] < 1.0
    assert pred_call["confidence"] < 1.0
    
    # TASK 12: Verification points
    assert pred_call["data_quality_state"] == "PARTIAL"
    assert "gate_state_transition_horizon_5: VALID -> PARTIAL" in caplog.text