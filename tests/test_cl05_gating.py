# tests/test_cl05_gating.py
import pytest
import datetime as dt
import logging
from unittest.mock import MagicMock, patch

from src.models import DecisionGate, DataQualityState, RiskGateStatus, SignalState

def test_gate_transitions_missing_features():
    initial_gate = DecisionGate(
        data_quality_state=DataQualityState.VALID, 
        risk_gate_status=RiskGateStatus.PASS, 
        decision_state=SignalState.NEUTRAL
    )
    
    missing_deps = ["oi_pressure", "smart_whale_pressure"]
    
    blocked_gate = initial_gate.block(
        reason=f"critical_features_missing: {','.join(missing_deps)}", 
        invalid=True, 
        missing_features=missing_deps
    )
    
    assert blocked_gate.risk_gate_status == RiskGateStatus.BLOCKED
    assert blocked_gate.decision_state == SignalState.NO_SIGNAL
    assert blocked_gate.validation_eligible is False

def test_gate_transitions_degraded():
    initial_gate = DecisionGate(
        data_quality_state=DataQualityState.VALID, 
        risk_gate_status=RiskGateStatus.PASS, 
        decision_state=SignalState.NEUTRAL
    )
    
    degraded_gate = initial_gate.degrade("non_critical_features_missing: dealer_charm", partial=True)
    
    assert degraded_gate.risk_gate_status == RiskGateStatus.DEGRADED
    assert degraded_gate.data_quality_state == DataQualityState.PARTIAL
    assert degraded_gate.validation_eligible is True  

def test_integration_premarket_missing_greeks(caplog):
    from src.ingest_engine import IngestionEngine

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
            "use_default_required_features": True,
            "emit_to_close_horizon": False,
            "horizon_critical_features": {}
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
        mock_mh.get_session_label.return_value = "PREMARKET"
        mock_mh.seconds_to_close.return_value = 3600
        mock_gmh.return_value = mock_mh

        mock_lep.return_value = {"plans": {"default": [
            {"name": "spot", "method": "GET", "path": "/api/stock/{ticker}/ohlc/1m"}
        ]}}

        mock_call = MagicMock()
        mock_call.method, mock_call.path = "GET", "/api/stock/{ticker}/ohlc/1m"
        
        mock_res = MagicMock()
        mock_res.requested_at_utc = dt.datetime.now(dt.timezone.utc).timestamp()
        mock_res.received_at_utc = mock_res.requested_at_utc + 0.1
        mock_res.status_code = 200
        mock_res.payload_hash = "hash"
        mock_res.payload_json = [{"t": mock_res.requested_at_utc, "close": 150.0}] 
        mock_res.retry_count = 0
        mock_res.error_type = None
        mock_res.error_message = None
        
        async def fake_fetch(*args, **kwargs):
            return [("AAPL", mock_call, "sig", {}, mock_res, None)]
        mock_fetch.side_effect = fake_fetch

        mock_db = MagicMock()
        mock_dbw_cls.return_value = mock_db
        mock_db.writer.return_value.__enter__.return_value = MagicMock()
        mock_db.get_payloads_by_event_ids.return_value = {"uuid1": mock_res.payload_json}

        engine = IngestionEngine(cfg=cfg, catalog_path="api_catalog.generated.yaml", config_path="dummy.yaml")
        
        with patch('src.ingest_engine.extract_all') as mock_extract:
            past_utc = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=15)).isoformat()
            valid_meta = {
                "source_endpoints": [], "freshness_state": "FRESH", "stale_age_min": 0, "na_reason": None, "details": {},
                "metric_lineage": {"effective_ts_utc": past_utc}
            }
            
            mock_extract.return_value = (
                [{"feature_key": "spot", "feature_value": 150.0, "meta_json": valid_meta}],
                []
            )
            
            with caplog.at_level(logging.WARNING):
                engine.run_cycle()
            
            assert "critical_feature_failed" in caplog.text
            assert "dealer_vanna" in caplog.text

def test_integration_horizon_aware_gating(caplog):
    from src.ingest_engine import IngestionEngine

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
            "use_default_required_features": False,
            "emit_to_close_horizon": False,
            "horizons_minutes": [5, 10],
            "horizon_critical_features": {
                "5": ["spot", "oi_pressure"],
                "10": ["spot"]
            },
            "horizon_weights": {
                "5": {"oi_pressure": 1.0},
                "10": {"oi_pressure": 1.0}
            }
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
        mock_mh.seconds_to_close.return_value = 0
        mock_gmh.return_value = mock_mh

        mock_lep.return_value = {"plans": {"default": [
            {"name": "spot", "method": "GET", "path": "/api/stock/{ticker}/ohlc/1m"}
        ]}}

        mock_call = MagicMock()
        mock_call.method, mock_call.path = "GET", "/api/stock/{ticker}/ohlc/1m"
        
        mock_res = MagicMock()
        mock_res.requested_at_utc = dt.datetime.now(dt.timezone.utc).timestamp()
        mock_res.received_at_utc = mock_res.requested_at_utc + 0.1
        mock_res.status_code = 200
        mock_res.payload_hash = "hash"
        mock_res.payload_json = [{"t": mock_res.requested_at_utc, "close": 150.0}] 
        mock_res.retry_count = 0
        mock_res.error_type = None
        mock_res.error_message = None
        
        async def fake_fetch(*args, **kwargs):
            return [("AAPL", mock_call, "sig", {}, mock_res, None)]
        mock_fetch.side_effect = fake_fetch

        mock_db = MagicMock()
        mock_dbw_cls.return_value = mock_db
        mock_db.writer.return_value.__enter__.return_value = MagicMock()
        mock_db.get_payloads_by_event_ids.return_value = {"uuid1": mock_res.payload_json}

        engine = IngestionEngine(cfg=cfg, catalog_path="api_catalog.generated.yaml", config_path="dummy.yaml")
        
        with patch('src.ingest_engine.extract_all') as mock_extract:
            past_utc = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=15)).isoformat()
            valid_meta = {
                "source_endpoints": [], "freshness_state": "FRESH", "stale_age_min": 0, "na_reason": None, "details": {},
                "metric_lineage": {"effective_ts_utc": past_utc}
            }
            
            mock_extract.return_value = (
                [{"feature_key": "spot", "feature_value": 150.0, "meta_json": valid_meta}],
                []
            )
            
            engine.run_cycle()
            
            pred_calls = mock_db.insert_prediction.call_args_list
            assert len(pred_calls) == 2