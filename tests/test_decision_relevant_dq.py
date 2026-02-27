# tests/test_decision_relevant_dq.py
import pytest
import datetime as dt
import logging
from unittest.mock import MagicMock, patch

import src.ingest_engine as ie_mod
from src.ingest_engine import IngestionEngine

@pytest.fixture
def mock_engine_env():
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
            "alignment_tolerance_sec": 900,
            "use_default_required_features": False,
            "emit_to_close_horizon": False, 
            "horizon_weights_source": "explicit",
            "horizons_minutes": [5],
            "horizon_critical_features": {"5": ["spot"]},
            "horizon_weights": {"5": {"spot": 1.0, "oi_pressure": 1.0}}
        }
    }
    return cfg

def _run_with_features(cfg, features, caplog, mock_fetch_len=10, mock_fetch_success=10):
    with patch.object(ie_mod, "get_market_hours") as mock_gmh, \
         patch.object(ie_mod, "fetch_all") as mock_fetch, \
         patch.object(ie_mod, "load_endpoint_plan") as mock_lep, \
         patch.object(ie_mod, "load_api_catalog"), \
         patch.object(ie_mod, "validate_plan_coverage"), \
         patch.object(ie_mod, "DbWriter") as mock_dbw_cls, \
         patch.object(ie_mod, "FileLock"):

        mock_mh = MagicMock()
        mock_mh.is_trading_day = True
        mock_mh.ingest_start_et = dt.datetime(2000, 1, 1, tzinfo=dt.timezone.utc)
        mock_mh.ingest_end_et = dt.datetime(2100, 1, 1, tzinfo=dt.timezone.utc)
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

        engine = IngestionEngine(cfg=cfg, catalog_path="api_catalog.generated.yaml", config_path="src/config/config.yaml")
        
        with patch.object(ie_mod, 'extract_all') as mock_extract:
            mock_extract.return_value = (features, [])
            with caplog.at_level(logging.INFO):
                engine.run_cycle()
                
        return mock_db

def test_dq_is_low_when_critical_missing_despite_high_endpoint_coverage(mock_engine_env, caplog):
    valid_meta = {"source_endpoints": [], "freshness_state": "FRESH", "stale_age_min": 0, "na_reason": None, "details": {}}
    past_utc = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=15)).isoformat()
    
    features = [
        {"feature_key": "oi_pressure", "feature_value": 1000.0, "meta_json": {**valid_meta, "metric_lineage": {"effective_ts_utc": past_utc}}}
    ]
    
    mock_db = _run_with_features(mock_engine_env, features, caplog, mock_fetch_len=10, mock_fetch_success=10)
    
    calls = [call[0][1] for call in mock_db.insert_prediction.call_args_list]
    pred_call = next(c for c in calls if c["horizon_kind"] == "FIXED" and c["horizon_minutes"] == 5)
    
    assert "spot_missing_or_invalid" in pred_call["meta_json"]["dq_reason_codes"]
    assert pred_call["meta_json"]["decision_dq"] == 0.5 
    assert pred_call["decision_state"] == "NO_SIGNAL"
    assert pred_call["data_quality_state"] == "INVALID"

def test_dq_acceptable_when_endpoints_fail_but_required_features_present(mock_engine_env, caplog):
    valid_meta = {"source_endpoints": [], "freshness_state": "FRESH", "stale_age_min": 0, "na_reason": None, "details": {}}
    past_utc = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=15)).isoformat()
    
    features = [
        {"feature_key": "spot", "feature_value": 150.0, "meta_json": {**valid_meta, "metric_lineage": {"effective_ts_utc": past_utc}}},
        {"feature_key": "oi_pressure", "feature_value": 1000.0, "meta_json": {**valid_meta, "metric_lineage": {"effective_ts_utc": past_utc}}}
    ]
    
    mock_db = _run_with_features(mock_engine_env, features, caplog, mock_fetch_len=10, mock_fetch_success=2)
    
    calls = [call[0][1] for call in mock_db.insert_prediction.call_args_list]
    pred_call = next(c for c in calls if c["horizon_kind"] == "FIXED" and c["horizon_minutes"] == 5)
    
    assert pred_call["meta_json"]["decision_dq"] == 1.0
    assert len(pred_call["meta_json"]["dq_reason_codes"]) == 0
    assert pred_call["decision_state"] != "NO_SIGNAL"
    assert pred_call["data_quality_state"] == "VALID"

def test_stale_features_reduce_dq(mock_engine_env, caplog):
    valid_meta = {"source_endpoints": [], "freshness_state": "STALE_CARRY", "stale_age_min": 30, "na_reason": None, "details": {}}
    past_utc = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=15)).isoformat()
    
    features = [
        {"feature_key": "spot", "feature_value": 150.0, "meta_json": {**valid_meta, "metric_lineage": {"effective_ts_utc": past_utc}}},
        {"feature_key": "oi_pressure", "feature_value": 1000.0, "meta_json": {**valid_meta, "metric_lineage": {"effective_ts_utc": past_utc}}}
    ]
    
    mock_db = _run_with_features(mock_engine_env, features, caplog)
    
    calls = [call[0][1] for call in mock_db.insert_prediction.call_args_list]
    pred_call = next(c for c in calls if c["horizon_kind"] == "FIXED" and c["horizon_minutes"] == 5)
    
    assert "spot_stale_carry_age_30m" in pred_call["meta_json"]["dq_reason_codes"]
    assert pred_call["meta_json"]["decision_dq"] < 1.0 
    assert pred_call["data_quality_state"] == "PARTIAL" 

def test_stale_non_critical_lowers_confidence(mock_engine_env, caplog):
    valid_meta_fresh = {"source_endpoints": [], "freshness_state": "FRESH", "stale_age_min": 0, "na_reason": None, "details": {}}
    valid_meta_stale = {"source_endpoints": [], "freshness_state": "STALE_CARRY", "stale_age_min": 15, "na_reason": None, "details": {}}
    past_utc = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=15)).isoformat()
    
    features = [
        {"feature_key": "spot", "feature_value": 150.0, "meta_json": {**valid_meta_fresh, "metric_lineage": {"effective_ts_utc": past_utc}}},
        {"feature_key": "oi_pressure", "feature_value": 1000.0, "meta_json": {**valid_meta_stale, "metric_lineage": {"effective_ts_utc": past_utc}}}
    ]
    
    mock_engine_env["validation"]["horizon_critical_features"] = {"5": ["spot"]}
    mock_engine_env["validation"]["horizon_weights"] = {"5": {"spot": 1.0, "oi_pressure": 1.0}}
    mock_engine_env["validation"]["use_default_required_features"] = False
    
    mock_db = _run_with_features(mock_engine_env, features, caplog)
    
    calls = [call[0][1] for call in mock_db.insert_prediction.call_args_list]
    pred_call = next(c for c in calls if c["horizon_kind"] == "FIXED" and c["horizon_minutes"] == 5)
    
    assert "oi_pressure_stale_carry_age_15m" in pred_call["meta_json"]["dq_reason_codes"]
    assert "spot_stale_carry_age_15m" not in pred_call["meta_json"]["dq_reason_codes"]
    assert pred_call["decision_state"] != "NO_SIGNAL"
    assert pred_call["meta_json"]["decision_dq"] < 1.0
    assert pred_call["confidence"] < 1.0
    assert pred_call["data_quality_state"] == "PARTIAL"

def test_explicit_horizon_contract_no_fallback(mock_engine_env, caplog):
    valid_meta = {"source_endpoints": [], "freshness_state": "FRESH", "stale_age_min": 0, "na_reason": None, "details": {}}
    past_utc = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=15)).isoformat()
    
    features = [
        {"feature_key": "oi_pressure", "feature_value": 1000.0, "meta_json": {**valid_meta, "metric_lineage": {"effective_ts_utc": past_utc}}}
    ]
    
    mock_engine_env["validation"]["horizon_critical_features"] = {"5": []}
    mock_engine_env["validation"]["horizon_weights"] = {"5": {"oi_pressure": 1.0}}
    mock_engine_env["validation"]["use_default_required_features"] = False
    
    mock_db = _run_with_features(mock_engine_env, features, caplog)
    
    calls = [call[0][1] for call in mock_db.insert_prediction.call_args_list]
    pred_call = next(c for c in calls if c["horizon_kind"] == "FIXED" and c["horizon_minutes"] == 5)
    
    assert "horizon_contract" in pred_call["meta_json"]
    contract = pred_call["meta_json"]["horizon_contract"]
    assert contract["use_default_required_features"] is False
    assert contract["resolved_critical_features"] == []

def test_invalid_horizon_contract_hard_fails(mock_engine_env, caplog):
    valid_meta = {"source_endpoints": [], "freshness_state": "FRESH", "stale_age_min": 0, "na_reason": None, "details": {}}
    past_utc = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=15)).isoformat()
    
    features = [
        {"feature_key": "oi_pressure", "feature_value": 1000.0, "meta_json": {**valid_meta, "metric_lineage": {"effective_ts_utc": past_utc}}}
    ]
    
    mock_engine_env["validation"]["horizon_critical_features"] = {"5": []}
    mock_engine_env["validation"]["horizon_weights"] = {"5": {}}
    mock_engine_env["validation"]["use_default_required_features"] = False
    
    mock_db = _run_with_features(mock_engine_env, features, caplog)
    
    calls = [call[0][1] for call in mock_db.insert_prediction.call_args_list]
    pred_call = next(c for c in calls if c["horizon_kind"] == "FIXED" and c["horizon_minutes"] == 5)
    
    assert pred_call["decision_state"] == "NO_SIGNAL"
    assert pred_call["risk_gate_status"] == "BLOCKED"
    assert "invalid_contract_no_targets_5" in pred_call["blocked_reasons"]