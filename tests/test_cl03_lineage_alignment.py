# tests/test_cl03_lineage_alignment.py
import pytest
import datetime as dt
import uuid
import logging
from typing import Optional
from unittest.mock import MagicMock, patch

import src.ingest_engine as ie_mod
from src.endpoint_truth import (
    resolve_effective_payload, 
    PayloadAssessment, 
    EndpointPayloadClass, 
    EmptyPayloadPolicy, 
    FreshnessState, 
    EndpointStateRow
)
from src.features import _build_meta, EndpointContext
from src.ingest_engine import IngestionEngine

def test_stale_carry_preserves_original_time():
    attempt_ts = dt.datetime(2026, 1, 1, 12, 0, 0, tzinfo=dt.timezone.utc)
    original_success_ts = dt.datetime(2026, 1, 1, 11, 45, 0, tzinfo=dt.timezone.utc)
    
    assessment = PayloadAssessment(
        payload_class=EndpointPayloadClass.ERROR, 
        changed=False, 
        missing_keys=[], 
        error_reason="HTTP_500", 
        empty_policy=EmptyPayloadPolicy.EMPTY_IS_DATA,
        is_empty=True, validator=None
    )
    
    prev_state = EndpointStateRow(
        last_success_event_id=str(uuid.uuid4()),
        last_success_ts_utc=original_success_ts,
        last_payload_hash="abcd",
        last_change_ts_utc=original_success_ts,
        last_change_event_id=str(uuid.uuid4())
    )
    
    resolved = resolve_effective_payload(
        current_event_id=str(uuid.uuid4()),
        current_ts_raw=attempt_ts,
        assessment=assessment,
        prev_state=prev_state,
        fallback_max_age_seconds=1800
    )
    
    assert resolved.freshness_state == FreshnessState.STALE_CARRY
    assert resolved.effective_ts_utc == original_success_ts
    assert resolved.effective_ts_utc != attempt_ts

def test_feature_extraction_lineage_propagation():
    eff_ts = dt.datetime(2026, 1, 1, 10, 0, 0, tzinfo=dt.timezone.utc)
    
    ctx = EndpointContext(
        endpoint_id=1,
        method="GET",
        path="/api/test",
        operation_id="test_op",
        signature="GET /api/test",
        used_event_id="mock-uuid",
        payload_class="SUCCESS_HAS_DATA",
        freshness_state="FRESH",
        stale_age_min=0,
        na_reason=None,
        endpoint_asof_ts_utc=dt.datetime.now(dt.timezone.utc),
        alignment_delta_sec=0,
        effective_ts_utc=eff_ts
    )
    
    meta = _build_meta(ctx, "test_extractor", {"metric_name": "test_metric"})
    assert meta["metric_lineage"]["effective_ts_utc"] == eff_ts.isoformat()

def test_alignment_gating_counters_and_status(caplog):
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
            "horizons_minutes": [5], 
            "alignment_tolerance_sec": 300,
            "use_default_required_features": False,
            "emit_to_close_horizon": False,
            "horizon_critical_features": {"5": ["spot", "net_gex_sign"]}  # STRICT CONTRACT
        }
    }

    with patch.object(ie_mod, "get_market_hours") as mock_gmh, \
         patch.object(ie_mod, "fetch_all") as mock_fetch, \
         patch.object(ie_mod, "load_endpoint_plan") as mock_lep, \
         patch.object(ie_mod, "load_api_catalog") as mock_lac, \
         patch.object(ie_mod, "validate_plan_coverage") as mock_vpc, \
         patch.object(ie_mod, "DbWriter") as mock_dbw_cls, \
         patch.object(ie_mod, "FileLock") as mock_fl, \
         patch.object(ie_mod, "extract_all") as mock_extract, \
         patch.object(ie_mod, "floor_to_interval") as mock_floor:
             
        mock_mh = MagicMock()
        mock_mh.is_trading_day = True
        mock_mh.ingest_start_et = dt.datetime(2000, 1, 1, tzinfo=dt.timezone.utc)
        mock_mh.ingest_end_et = dt.datetime(2100, 1, 1, tzinfo=dt.timezone.utc)
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
        
        from src.scheduler import ET
        fixed_asof = dt.datetime(2026, 1, 1, 12, 0, tzinfo=dt.timezone.utc)
        mock_floor.return_value = fixed_asof.astimezone(ET)
        
        stale_ts = fixed_asof - dt.timedelta(seconds=1000)
        f1_ts = fixed_asof - dt.timedelta(seconds=10)
        
        valid_meta = {"source_endpoints": [], "freshness_state": "FRESH", "stale_age_min": 0, "na_reason": None, "details": {}}
        
        f1 = {
            "feature_key": "spot", "feature_value": 150.0, 
            "meta_json": {**valid_meta, "metric_lineage": {"effective_ts_utc": f1_ts.isoformat()}}
        }
        f2 = {
            "feature_key": "net_gex_sign", "feature_value": 1.0, 
            "meta_json": {**valid_meta, "metric_lineage": {"effective_ts_utc": stale_ts.isoformat()}}
        }
        
        mock_extract.return_value = ([f1, f2], [])
        
        engine = IngestionEngine(cfg=cfg, catalog_path="api_catalog.generated.yaml", config_path="dummy.yaml")
        
        with caplog.at_level(logging.WARNING):
            engine.run_cycle()
            
        assert "alignment_violation" in caplog.text
        assert "net_gex_sign" in caplog.text
        
        calls = [c[0][1] for c in mock_db.insert_prediction.call_args_list]
        pred_call = next(c for c in calls if c["horizon_kind"] == "FIXED" and c["horizon_minutes"] == 5)
        
        assert pred_call["alignment_status"] == "MISALIGNED"
        assert pred_call["decision_state"] == "NO_SIGNAL"

def test_aligned_fixture_produces_aligned_status():
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
            "horizons_minutes": [5], 
            "alignment_tolerance_sec": 300,
            "use_default_required_features": False,
            "emit_to_close_horizon": False,
            "horizon_critical_features": {}
        }
    }

    with patch.object(ie_mod, "get_market_hours") as mock_gmh, \
         patch.object(ie_mod, "fetch_all") as mock_fetch, \
         patch.object(ie_mod, "load_endpoint_plan") as mock_lep, \
         patch.object(ie_mod, "load_api_catalog") as mock_lac, \
         patch.object(ie_mod, "validate_plan_coverage") as mock_vpc, \
         patch.object(ie_mod, "DbWriter") as mock_dbw_cls, \
         patch.object(ie_mod, "FileLock") as mock_fl, \
         patch.object(ie_mod, "extract_all") as mock_extract, \
         patch.object(ie_mod, "floor_to_interval") as mock_floor:
             
        mock_mh = MagicMock()
        mock_mh.is_trading_day = True
        mock_mh.ingest_start_et = dt.datetime(2000, 1, 1, tzinfo=dt.timezone.utc)
        mock_mh.ingest_end_et = dt.datetime(2100, 1, 1, tzinfo=dt.timezone.utc)
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
        
        from src.scheduler import ET
        fixed_asof = dt.datetime(2026, 1, 1, 12, 0, tzinfo=dt.timezone.utc)
        mock_floor.return_value = fixed_asof.astimezone(ET)
        
        aligned_ts_1 = fixed_asof - dt.timedelta(seconds=10)
        aligned_ts_2 = fixed_asof - dt.timedelta(seconds=20)
        
        valid_meta = {"source_endpoints": [], "freshness_state": "FRESH", "stale_age_min": 0, "na_reason": None, "details": {}}
        
        f1 = {
            "feature_key": "spot", "feature_value": 150.0, 
            "meta_json": {**valid_meta, "metric_lineage": {"effective_ts_utc": aligned_ts_1.isoformat()}}
        }
        f2 = {
            "feature_key": "net_gex_sign", "feature_value": 1.0, 
            "meta_json": {**valid_meta, "metric_lineage": {"effective_ts_utc": aligned_ts_2.isoformat()}}
        }
        
        mock_extract.return_value = ([f1, f2], [])
        
        engine = IngestionEngine(cfg=cfg, catalog_path="api_catalog.generated.yaml", config_path="dummy.yaml")
        engine.run_cycle()
            
        calls = [c[0][1] for c in mock_db.insert_prediction.call_args_list]
        pred_call = next(c for c in calls if c["horizon_kind"] == "FIXED" and c["horizon_minutes"] == 5)
        
        assert pred_call["alignment_status"] == "ALIGNED"
        assert pred_call["source_ts_min_utc"] == aligned_ts_2
        assert pred_call["source_ts_max_utc"] == aligned_ts_1