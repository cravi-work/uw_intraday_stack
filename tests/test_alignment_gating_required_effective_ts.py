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
            "horizon_critical_features": {"5": ["spot", "oi_pressure"]},
            "horizon_weights": {"5": {"spot": 1.0, "oi_pressure": 1.0, "iv_rank": 0.5}}
        }
    }
    return cfg

def _run_with_features(cfg, features, caplog):
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

        async def fake_fetch(*args, **kwargs): return []
        mock_fetch.side_effect = fake_fetch

        mock_db = MagicMock()
        mock_dbw_cls.return_value = mock_db
        mock_db.writer.return_value.__enter__.return_value = MagicMock()
        mock_db.get_payloads_by_event_ids.return_value = {}

        engine = IngestionEngine(cfg=cfg, catalog_path="dummy.yaml", config_path="dummy.yaml")
        
        with patch('src.ingest_engine.extract_all') as mock_extract:
            mock_extract.return_value = (features, [])
            with caplog.at_level(logging.WARNING):
                engine.run_cycle()
                
        return mock_db

def test_missing_effective_ts_critical_feature(mock_engine_env, caplog):
    valid_meta = {"source_endpoints": [], "freshness_state": "FRESH", "stale_age_min": 0, "na_reason": None, "details": {}}
    now_utc = dt.datetime.now(dt.timezone.utc).isoformat()
    
    features = [
        {"feature_key": "spot", "feature_value": 150.0, "meta_json": {**valid_meta, "metric_lineage": {}}},
        {"feature_key": "oi_pressure", "feature_value": 1000.0, "meta_json": {**valid_meta, "metric_lineage": {"effective_ts_utc": now_utc}}}
    ]
    
    mock_db = _run_with_features(mock_engine_env, features, caplog)
    
    assert "feature_missing_effective_ts" in caplog.text
    
    pred_call = mock_db.insert_prediction.call_args[0][1]
    assert pred_call["decision_state"] == "NO_SIGNAL"
    assert pred_call["risk_gate_status"] == "BLOCKED"

def test_misaligned_effective_ts_critical_feature(mock_engine_env, caplog):
    valid_meta = {"source_endpoints": [], "freshness_state": "FRESH", "stale_age_min": 0, "na_reason": None, "details": {}}
    now_utc = dt.datetime.now(dt.timezone.utc)
    stale_ts = (now_utc - dt.timedelta(seconds=2000)).isoformat() 
    
    features = [
        {"feature_key": "spot", "feature_value": 150.0, "meta_json": {**valid_meta, "metric_lineage": {"effective_ts_utc": now_utc.isoformat()}}},
        {"feature_key": "oi_pressure", "feature_value": 1000.0, "meta_json": {**valid_meta, "metric_lineage": {"effective_ts_utc": stale_ts}}}
    ]
    
    mock_db = _run_with_features(mock_engine_env, features, caplog)
    
    assert "alignment_violation" in caplog.text
    
    pred_call = mock_db.insert_prediction.call_args[0][1]
    assert pred_call["decision_state"] == "NO_SIGNAL"
    assert pred_call["risk_gate_status"] == "BLOCKED"

def test_missing_ts_non_critical_feature_degrades(mock_engine_env, caplog):
    valid_meta = {"source_endpoints": [], "freshness_state": "FRESH", "stale_age_min": 0, "na_reason": None, "details": {}}
    now_utc = dt.datetime.now(dt.timezone.utc).isoformat()
    
    features = [
        {"feature_key": "spot", "feature_value": 150.0, "meta_json": {**valid_meta, "metric_lineage": {"effective_ts_utc": now_utc}}},
        {"feature_key": "oi_pressure", "feature_value": 1000.0, "meta_json": {**valid_meta, "metric_lineage": {"effective_ts_utc": now_utc}}},
        {"feature_key": "iv_rank", "feature_value": 0.5, "meta_json": {**valid_meta, "metric_lineage": {}}} 
    ]
    
    mock_db = _run_with_features(mock_engine_env, features, caplog)
    
    assert "feature_missing_effective_ts" in caplog.text
    
    pred_call = mock_db.insert_prediction.call_args[0][1]
    assert pred_call["decision_state"] != "NO_SIGNAL" 
    assert pred_call["risk_gate_status"] == "DEGRADED"

def test_naive_timestamp_rejected(mock_engine_env, caplog):
    """
    EVIDENCE: Explicitly verify that an ISO string missing timezone offset (+00:00) 
    is intercepted and treated as missing rather than silently converted.
    """
    valid_meta = {"source_endpoints": [], "freshness_state": "FRESH", "stale_age_min": 0, "na_reason": None, "details": {}}
    naive_ts = dt.datetime.utcnow().isoformat() # Lacks tzinfo
    
    features = [
        {"feature_key": "spot", "feature_value": 150.0, "meta_json": {**valid_meta, "metric_lineage": {"effective_ts_utc": naive_ts}}},
        {"feature_key": "oi_pressure", "feature_value": 1000.0, "meta_json": {**valid_meta, "metric_lineage": {"effective_ts_utc": naive_ts}}}
    ]
    
    mock_db = _run_with_features(mock_engine_env, features, caplog)
    
    assert "feature_invalid_effective_ts (naive timezone)" in caplog.text
    
    pred_call = mock_db.insert_prediction.call_args[0][1]
    assert pred_call["decision_state"] == "NO_SIGNAL"
    assert pred_call["meta_json"]["alignment_diagnostics"]["excluded_missing_ts_count"] == 2
    assert "spot" in pred_call["meta_json"]["alignment_diagnostics"]["missing_ts_keys"]

def test_misaligned_feature_does_not_pollute_ts_min_max(mock_engine_env, caplog):
    """
    EVIDENCE: Misaligned features are excluded entirely from source_ts_min and source_ts_max calculations.
    """
    valid_meta = {"source_endpoints": [], "freshness_state": "FRESH", "stale_age_min": 0, "na_reason": None, "details": {}}
    now_utc = dt.datetime.now(dt.timezone.utc)
    aligned_ts = now_utc.isoformat()
    stale_ts = (now_utc - dt.timedelta(seconds=2000)).isoformat()
    
    features = [
        {"feature_key": "spot", "feature_value": 150.0, "meta_json": {**valid_meta, "metric_lineage": {"effective_ts_utc": aligned_ts}}},
        {"feature_key": "iv_rank", "feature_value": 0.5, "meta_json": {**valid_meta, "metric_lineage": {"effective_ts_utc": stale_ts}}}
    ]
    
    mock_engine_env["validation"]["horizon_critical_features"] = {"5": ["spot"]}
    
    mock_db = _run_with_features(mock_engine_env, features, caplog)
    
    pred_call = mock_db.insert_prediction.call_args[0][1]
    
    # Assert window is clean (matches 'now' exactly, ignoring the older stale_ts)
    assert pred_call["source_ts_min_utc"] == now_utc
    assert pred_call["source_ts_max_utc"] == now_utc
    
    # Assert alignment diagnostics properly caught it
    assert pred_call["meta_json"]["alignment_diagnostics"]["excluded_misaligned_count"] == 1
    assert any("iv_rank_delta_" in k for k in pred_call["meta_json"]["alignment_diagnostics"]["misaligned_keys"])