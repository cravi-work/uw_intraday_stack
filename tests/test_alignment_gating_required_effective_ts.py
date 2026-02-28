# tests/test_alignment_gating_required_effective_ts.py
import pytest
import datetime as dt
import logging
import copy
from unittest.mock import MagicMock, patch

import src.ingest_engine as ie_mod
from src.ingest_engine import IngestionEngine
from src.scheduler import ET

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
            "invalid_after_minutes": 60,
            "tolerance_minutes": 10,
            "max_horizon_drift_minutes": 10,
            "flat_threshold_pct": 0.001,
            "fallback_max_age_minutes": 15,
            "use_default_required_features": False,
            "emit_to_close_horizon": True,
            "horizon_weights_source": "explicit",
            "horizons_minutes": [5],
            "horizon_critical_features": {"5": ["spot", "oi_pressure"], "to_close": ["spot", "oi_pressure"]},
            "horizon_weights": {"5": {"spot": 1.0, "oi_pressure": 1.0, "iv_rank": 0.5}, "to_close": {"spot": 1.0, "oi_pressure": 1.0}}
        }
    }
    return cfg

def _run_with_features(cfg, features, caplog, fixed_asof=None):
    if fixed_asof is None:
        fixed_asof = dt.datetime(2026, 1, 1, 12, 0, tzinfo=dt.timezone.utc)
        
    with patch.object(ie_mod, "get_market_hours") as mock_gmh, \
         patch.object(ie_mod, "fetch_all") as mock_fetch, \
         patch.object(ie_mod, "load_endpoint_plan") as mock_lep, \
         patch.object(ie_mod, "load_api_catalog"), \
         patch.object(ie_mod, "validate_plan_coverage"), \
         patch.object(ie_mod, "DbWriter") as mock_dbw_cls, \
         patch.object(ie_mod, "FileLock"), \
         patch.object(ie_mod, "extract_all") as mock_extract, \
         patch.object(ie_mod, "floor_to_interval") as mock_floor:

        mock_mh = MagicMock()
        mock_mh.is_trading_day = True
        mock_mh.ingest_start_et = fixed_asof - dt.timedelta(hours=1)
        mock_mh.ingest_end_et = fixed_asof + dt.timedelta(hours=1)
        mock_mh.get_session_label.return_value = "RTH"
        mock_mh.seconds_to_close.return_value = 3600
        mock_gmh.return_value = mock_mh

        mock_floor.return_value = fixed_asof.astimezone(ET)
        mock_lep.return_value = {"plans": {"default": []}}

        async def fake_fetch(*args, **kwargs): return []
        mock_fetch.side_effect = fake_fetch

        mock_db = MagicMock()
        mock_dbw_cls.return_value = mock_db
        mock_db.writer.return_value.__enter__.return_value = MagicMock()
        mock_db.get_payloads_by_event_ids.return_value = {}

        mock_extract.return_value = (features, [])

        engine = IngestionEngine(cfg=cfg, catalog_path="api_catalog.generated.yaml", config_path="src/config/config.yaml")
        
        with caplog.at_level(logging.WARNING):
            engine.run_cycle()
                
        return mock_db

def test_clamp_future_ts_rewrites_lineage_meta(mock_engine_env, caplog):
    """
    Task 7 Proof: A future timestamp within cadence drift is clamped, 
    AND its lineage JSON is explicitly rewritten to match asof_utc before database storage.
    """
    fixed_asof = dt.datetime(2026, 1, 1, 12, 0, tzinfo=dt.timezone.utc)
    valid_meta = {"source_endpoints": [], "freshness_state": "FRESH", "stale_age_min": 0, "na_reason": None, "details": {}}
    
    # 2 minutes future drift (within 5 min cadence) -> Triggers Clamp
    future_utc = (fixed_asof + dt.timedelta(minutes=2)).isoformat()
    
    f1 = {"feature_key": "spot", "feature_value": 150.0, "meta_json": copy.deepcopy(valid_meta)}
    f1["meta_json"]["metric_lineage"] = {"effective_ts_utc": future_utc}
    f2 = {"feature_key": "oi_pressure", "feature_value": 1000.0, "meta_json": copy.deepcopy(valid_meta)}
    f2["meta_json"]["metric_lineage"] = {"effective_ts_utc": fixed_asof.isoformat()}
    
    features = [f1, f2]
    
    mock_db = _run_with_features(mock_engine_env, features, caplog, fixed_asof)
    
    calls = [call[0][1] for call in mock_db.insert_prediction.call_args_list]
    pred_call = next(c for c in calls if c["horizon_kind"] == "FIXED" and c["horizon_minutes"] == 5)
    
    assert pred_call["decision_state"] != "NO_SIGNAL"
    assert pred_call["alignment_status"] == "ALIGNED"
    assert pred_call["meta_json"]["alignment_diagnostics"]["normalized_future_ts_count"] == 1
    
    # Verify the actual list sent to DB had its lineage mutated identically
    inserted_features = mock_db.insert_features.call_args[0][2]
    spot_feat = next(f for f in inserted_features if f["feature_key"] == "spot")
    
    assert spot_feat["meta_json"]["metric_lineage"]["effective_ts_utc"] == fixed_asof.isoformat()
    assert spot_feat["meta_json"]["details"].get("clamped_future_ts") is True

def test_missing_effective_ts_critical_feature(mock_engine_env, caplog):
    fixed_asof = dt.datetime(2026, 1, 1, 12, 0, tzinfo=dt.timezone.utc)
    valid_meta = {"source_endpoints": [], "freshness_state": "FRESH", "stale_age_min": 0, "na_reason": None, "details": {}}
    past_utc = (fixed_asof - dt.timedelta(minutes=15)).isoformat()
    
    features = [
        {"feature_key": "spot", "feature_value": 150.0, "meta_json": {**valid_meta, "metric_lineage": {}}},
        {"feature_key": "oi_pressure", "feature_value": 1000.0, "meta_json": {**valid_meta, "metric_lineage": {"effective_ts_utc": past_utc}}}
    ]
    
    mock_db = _run_with_features(mock_engine_env, features, caplog, fixed_asof)
    
    assert "feature_missing_effective_ts" in caplog.text
    
    calls = [call[0][1] for call in mock_db.insert_prediction.call_args_list]
    pred_call = next(c for c in calls if c["horizon_kind"] == "FIXED" and c["horizon_minutes"] == 5)
    
    assert pred_call["decision_state"] == "NO_SIGNAL"
    assert pred_call["risk_gate_status"] == "BLOCKED"

def test_misaligned_effective_ts_critical_feature(mock_engine_env, caplog):
    fixed_asof = dt.datetime(2026, 1, 1, 12, 0, tzinfo=dt.timezone.utc)
    valid_meta = {"source_endpoints": [], "freshness_state": "FRESH", "stale_age_min": 0, "na_reason": None, "details": {}}
    past_utc = (fixed_asof - dt.timedelta(minutes=15)).isoformat()
    stale_ts = (fixed_asof - dt.timedelta(seconds=2000)).isoformat() 
    
    features = [
        {"feature_key": "spot", "feature_value": 150.0, "meta_json": {**valid_meta, "metric_lineage": {"effective_ts_utc": past_utc}}},
        {"feature_key": "oi_pressure", "feature_value": 1000.0, "meta_json": {**valid_meta, "metric_lineage": {"effective_ts_utc": stale_ts}}}
    ]
    
    mock_db = _run_with_features(mock_engine_env, features, caplog, fixed_asof)
    
    assert "alignment_violation" in caplog.text
    
    calls = [call[0][1] for call in mock_db.insert_prediction.call_args_list]
    pred_call = next(c for c in calls if c["horizon_kind"] == "FIXED" and c["horizon_minutes"] == 5)
    
    assert pred_call["decision_state"] == "NO_SIGNAL"
    assert pred_call["risk_gate_status"] == "BLOCKED"

def test_missing_ts_non_critical_feature_degrades(mock_engine_env, caplog):
    fixed_asof = dt.datetime(2026, 1, 1, 12, 0, tzinfo=dt.timezone.utc)
    valid_meta = {"source_endpoints": [], "freshness_state": "FRESH", "stale_age_min": 0, "na_reason": None, "details": {}}
    past_utc = (fixed_asof - dt.timedelta(minutes=15)).isoformat()
    
    features = [
        {"feature_key": "spot", "feature_value": 150.0, "meta_json": {**valid_meta, "metric_lineage": {"effective_ts_utc": past_utc}}},
        {"feature_key": "oi_pressure", "feature_value": 1000.0, "meta_json": {**valid_meta, "metric_lineage": {"effective_ts_utc": past_utc}}},
        {"feature_key": "iv_rank", "feature_value": 0.5, "meta_json": {**valid_meta, "metric_lineage": {}}} 
    ]
    
    mock_db = _run_with_features(mock_engine_env, features, caplog, fixed_asof)
    
    assert "feature_missing_effective_ts" in caplog.text
    
    calls = [call[0][1] for call in mock_db.insert_prediction.call_args_list]
    pred_call = next(c for c in calls if c["horizon_kind"] == "FIXED" and c["horizon_minutes"] == 5)
    
    assert pred_call["decision_state"] != "NO_SIGNAL" 
    assert pred_call["risk_gate_status"] == "DEGRADED"

def test_naive_timestamp_rejected(mock_engine_env, caplog):
    fixed_asof = dt.datetime(2026, 1, 1, 12, 0, tzinfo=dt.timezone.utc)
    valid_meta = {"source_endpoints": [], "freshness_state": "FRESH", "stale_age_min": 0, "na_reason": None, "details": {}}
    naive_ts = (fixed_asof.replace(tzinfo=None) - dt.timedelta(minutes=15)).isoformat() # Lacks tzinfo
    
    features = [
        {"feature_key": "spot", "feature_value": 150.0, "meta_json": {**valid_meta, "metric_lineage": {"effective_ts_utc": naive_ts}}},
        {"feature_key": "oi_pressure", "feature_value": 1000.0, "meta_json": {**valid_meta, "metric_lineage": {"effective_ts_utc": naive_ts}}}
    ]
    
    mock_db = _run_with_features(mock_engine_env, features, caplog, fixed_asof)
    
    assert "feature_invalid_effective_ts (naive timezone)" in caplog.text
    
    calls = [call[0][1] for call in mock_db.insert_prediction.call_args_list]
    pred_call = next(c for c in calls if c["horizon_kind"] == "FIXED" and c["horizon_minutes"] == 5)
    
    assert pred_call["decision_state"] == "NO_SIGNAL"
    assert pred_call["meta_json"]["alignment_diagnostics"]["excluded_missing_ts_count"] == 2
    assert "spot" in pred_call["meta_json"]["alignment_diagnostics"]["missing_ts_keys"]

def test_misaligned_feature_does_not_pollute_ts_min_max(mock_engine_env, caplog):
    fixed_asof = dt.datetime(2026, 1, 1, 12, 0, tzinfo=dt.timezone.utc)
    valid_meta = {"source_endpoints": [], "freshness_state": "FRESH", "stale_age_min": 0, "na_reason": None, "details": {}}
    aligned_ts_dt = fixed_asof - dt.timedelta(minutes=15)
    aligned_ts = aligned_ts_dt.isoformat()
    stale_ts = (fixed_asof - dt.timedelta(seconds=2000)).isoformat()
    
    features = [
        {"feature_key": "spot", "feature_value": 150.0, "meta_json": {**valid_meta, "metric_lineage": {"effective_ts_utc": aligned_ts}}},
        {"feature_key": "iv_rank", "feature_value": 0.5, "meta_json": {**valid_meta, "metric_lineage": {"effective_ts_utc": stale_ts}}}
    ]
    
    mock_engine_env["validation"]["horizon_critical_features"]["5"] = ["spot"]
    mock_engine_env["validation"]["horizon_weights"]["5"] = {"spot": 1.0, "iv_rank": 0.5}
    
    mock_db = _run_with_features(mock_engine_env, features, caplog, fixed_asof)
    
    calls = [call[0][1] for call in mock_db.insert_prediction.call_args_list]
    pred_call = next(c for c in calls if c["horizon_kind"] == "FIXED" and c["horizon_minutes"] == 5)
    
    assert pred_call["source_ts_min_utc"] == aligned_ts_dt
    assert pred_call["source_ts_max_utc"] == aligned_ts_dt
    
    assert pred_call["meta_json"]["alignment_diagnostics"]["excluded_misaligned_count"] == 1
    assert any("iv_rank_delta_" in k for k in pred_call["meta_json"]["alignment_diagnostics"]["misaligned_keys"])

def test_future_timestamp_within_tolerance_rejected(mock_engine_env, caplog):
    fixed_asof = dt.datetime(2026, 1, 1, 12, 0, tzinfo=dt.timezone.utc)
    valid_meta = {"source_endpoints": [], "freshness_state": "FRESH", "stale_age_min": 0, "na_reason": None, "details": {}}
    
    future_utc = (fixed_asof + dt.timedelta(minutes=10)).isoformat()
    past_utc = (fixed_asof - dt.timedelta(minutes=15)).isoformat()
    
    features = [
        {"feature_key": "spot", "feature_value": 150.0, "meta_json": {**valid_meta, "metric_lineage": {"effective_ts_utc": past_utc}}},
        {"feature_key": "oi_pressure", "feature_value": 1000.0, "meta_json": {**valid_meta, "metric_lineage": {"effective_ts_utc": future_utc}}}
    ]
    
    mock_db = _run_with_features(mock_engine_env, features, caplog, fixed_asof)
    
    assert "future_ts_violation" in caplog.text
    
    calls = [call[0][1] for call in mock_db.insert_prediction.call_args_list]
    pred_call = next(c for c in calls if c["horizon_kind"] == "FIXED" and c["horizon_minutes"] == 5)
    
    assert pred_call["decision_state"] == "NO_SIGNAL"
    assert pred_call["risk_gate_status"] == "BLOCKED"
    assert "oi_pressure_future_ts" in pred_call["blocked_reasons"][0]
    assert pred_call["alignment_status"] == "MISALIGNED"
    assert pred_call["meta_json"]["alignment_diagnostics"]["excluded_future_ts_count"] == 1
    assert "oi_pressure" in pred_call["meta_json"]["alignment_diagnostics"]["future_ts_keys"]

def test_exact_boundary_timestamp_accepted(mock_engine_env, caplog):
    fixed_asof = dt.datetime(2026, 1, 1, 12, 0, tzinfo=dt.timezone.utc)
    valid_meta = {"source_endpoints": [], "freshness_state": "FRESH", "stale_age_min": 0, "na_reason": None, "details": {}}
    
    features = [
        {"feature_key": "spot", "feature_value": 150.0, "meta_json": {**valid_meta, "metric_lineage": {"effective_ts_utc": fixed_asof.isoformat()}}},
        {"feature_key": "oi_pressure", "feature_value": 1000.0, "meta_json": {**valid_meta, "metric_lineage": {"effective_ts_utc": fixed_asof.isoformat()}}}
    ]
    
    mock_db = _run_with_features(mock_engine_env, features, caplog, fixed_asof)
        
    calls = [call[0][1] for call in mock_db.insert_prediction.call_args_list]
    pred_call = next(c for c in calls if c["horizon_kind"] == "FIXED" and c["horizon_minutes"] == 5)
    
    assert pred_call["decision_state"] != "NO_SIGNAL"
    assert pred_call["alignment_status"] == "ALIGNED"
    assert pred_call["meta_json"]["alignment_diagnostics"]["excluded_future_ts_count"] == 0

def test_to_close_and_fixed_horizons_coexist(mock_engine_env, caplog):
    fixed_asof = dt.datetime(2026, 1, 1, 12, 0, tzinfo=dt.timezone.utc)
    valid_meta = {"source_endpoints": [], "freshness_state": "FRESH", "stale_age_min": 0, "na_reason": None, "details": {}}
    
    features = [
        {"feature_key": "spot", "feature_value": 150.0, "meta_json": {**valid_meta, "metric_lineage": {"effective_ts_utc": fixed_asof.isoformat()}}},
        {"feature_key": "oi_pressure", "feature_value": 1000.0, "meta_json": {**valid_meta, "metric_lineage": {"effective_ts_utc": fixed_asof.isoformat()}}}
    ]
    
    mock_db = _run_with_features(mock_engine_env, features, caplog, fixed_asof)
        
    calls = [call[0][1] for call in mock_db.insert_prediction.call_args_list]
    assert len(calls) == 2
    
    fixed_pred = next(c for c in calls if c["horizon_kind"] == "FIXED" and c["horizon_minutes"] == 5)
    close_pred = next(c for c in calls if c["horizon_kind"] == "TO_CLOSE" and c["horizon_minutes"] == 0)
    
    assert fixed_pred["decision_state"] != "NO_SIGNAL"
    assert close_pred["decision_state"] != "NO_SIGNAL"
    assert close_pred["horizon_seconds"] == 3600