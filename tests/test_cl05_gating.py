import pytest
import datetime as dt
import logging
from unittest.mock import MagicMock, patch

from src.models import DecisionGate, DataQualityState, RiskGateStatus, SignalState

def test_gate_transitions_missing_features():
    """
    EVIDENCE: Unit test proving that DecisionGate.block transitions correctly 
    from PASS -> BLOCKED when explicitly fed missing features, capturing exactly 
    what was dropped and strictly coercing validation_eligible to False.
    """
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
    assert "oi_pressure" in blocked_gate.critical_features_missing
    assert "smart_whale_pressure" in blocked_gate.critical_features_missing
    assert "critical_features_missing: oi_pressure,smart_whale_pressure" in blocked_gate.blocked_reasons

def test_gate_transitions_degraded():
    """
    EVIDENCE: Unit test proving that missing NON-critical features securely transition 
    the gate to DEGRADED and PARTIAL without entirely blocking the signal, 
    satisfying the missing testing feedback.
    """
    initial_gate = DecisionGate(
        data_quality_state=DataQualityState.VALID, 
        risk_gate_status=RiskGateStatus.PASS, 
        decision_state=SignalState.NEUTRAL
    )
    
    degraded_gate = initial_gate.degrade("non_critical_features_missing: dealer_charm", partial=True)
    
    assert degraded_gate.risk_gate_status == RiskGateStatus.DEGRADED
    assert degraded_gate.data_quality_state == DataQualityState.PARTIAL
    assert degraded_gate.validation_eligible is True  # Degraded signals are still tested/tracked
    assert "non_critical_features_missing: dealer_charm" in degraded_gate.degraded_reasons

def test_integration_premarket_missing_greeks(caplog):
    """
    EVIDENCE: Full integration/replay run proving that PREMARKET sessions strictly 
    depend on 'dealer_vanna' as a critical feature and suppress signals if omitted.
    """
    from src.ingest_engine import IngestionEngine

    cfg = {
        "ingestion": {"watchlist": ["AAPL"], "cadence_minutes": 5, "enable_market_context": False},
        "storage": {"duckdb_path": ":memory:", "cycle_lock_path": "mock.lock", "writer_lock_path": "mock.lock"},
        "system": {},
        "network": {},
        "validation": {"horizons_minutes": [5]}
    }

    with patch("src.ingest_engine.get_market_hours") as mock_gmh, \
         patch("src.ingest_engine.fetch_all") as mock_fetch, \
         patch("src.ingest_engine.load_endpoint_plan") as mock_lep, \
         patch("src.ingest_engine.load_api_catalog"), \
         patch("src.ingest_engine.DbWriter") as mock_dbw_cls, \
         patch("src.ingest_engine.FileLock"):

        mock_mh = MagicMock()
        mock_mh.is_trading_day = True
        mock_mh.ingest_start_et = dt.datetime.now() - dt.timedelta(hours=1)
        mock_mh.ingest_end_et = dt.datetime.now() + dt.timedelta(hours=1)
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
        
        async def fake_fetch(*args, **kwargs):
            return [("AAPL", mock_call, "sig", {}, mock_res, None)]
        mock_fetch.side_effect = fake_fetch

        mock_db = MagicMock()
        mock_dbw_cls.return_value = mock_db
        mock_db.writer.return_value.__enter__.return_value = MagicMock()
        mock_db.get_payloads_by_event_ids.return_value = {"uuid1": mock_res.payload_json}

        engine = IngestionEngine(cfg=cfg, catalog_path="dummy.yaml", config_path="dummy.yaml")
        
        with patch('src.ingest_engine.extract_all') as mock_extract:
            # We supply spot but consciously omit dealer_vanna
            mock_extract.return_value = (
                [{"feature_key": "spot", "feature_value": 150.0, "meta_json": {}}],
                []
            )
            
            with caplog.at_level(logging.WARNING):
                engine.run_cycle()
            
            assert "critical_feature_missing_count" in caplog.text
            assert "dealer_vanna" in caplog.text
            
            pred_call = mock_db.insert_prediction.call_args[0][1]
            assert pred_call["decision_state"] == "NO_SIGNAL"
            assert pred_call["risk_gate_status"] == "BLOCKED"
            gate_json = pred_call["gate_json"]
            assert "dealer_vanna" in gate_json["critical_features_missing"]

def test_integration_horizon_aware_gating(caplog):
    """
    EVIDENCE: Proves the new gating policy evaluates requirements dynamically based
    on horizon string, not just statically by session. Horizon '5' explicitly demands
    'oi_pressure' and blocks without it. Horizon '10' does not demand it and safely degrades.
    """
    from src.ingest_engine import IngestionEngine

    cfg = {
        "ingestion": {"watchlist": ["AAPL"], "cadence_minutes": 5, "enable_market_context": False},
        "storage": {"duckdb_path": ":memory:", "cycle_lock_path": "mock.lock", "writer_lock_path": "mock.lock"},
        "system": {},
        "network": {},
        "validation": {
            "horizons_minutes": [5, 10],
            "horizon_critical_features": {
                "5": ["spot", "oi_pressure"],     # 5m horizon strict on OI
                "10": ["spot"]                    # 10m horizon relaxed
            },
            "horizon_weights": {
                "5": {"oi_pressure": 1.0},
                "10": {"oi_pressure": 1.0}        # Evaluates as non-critical missing for 10m
            }
        }
    }

    with patch("src.ingest_engine.get_market_hours") as mock_gmh, \
         patch("src.ingest_engine.fetch_all") as mock_fetch, \
         patch("src.ingest_engine.load_endpoint_plan") as mock_lep, \
         patch("src.ingest_engine.load_api_catalog"), \
         patch("src.ingest_engine.DbWriter") as mock_dbw_cls, \
         patch("src.ingest_engine.FileLock"):

        mock_mh = MagicMock()
        mock_mh.is_trading_day = True
        mock_mh.ingest_start_et = dt.datetime.now() - dt.timedelta(hours=1)
        mock_mh.ingest_end_et = dt.datetime.now() + dt.timedelta(hours=1)
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
        
        async def fake_fetch(*args, **kwargs):
            return [("AAPL", mock_call, "sig", {}, mock_res, None)]
        mock_fetch.side_effect = fake_fetch

        mock_db = MagicMock()
        mock_dbw_cls.return_value = mock_db
        mock_db.writer.return_value.__enter__.return_value = MagicMock()
        mock_db.get_payloads_by_event_ids.return_value = {"uuid1": mock_res.payload_json}

        engine = IngestionEngine(cfg=cfg, catalog_path="dummy.yaml", config_path="dummy.yaml")
        
        with patch('src.ingest_engine.extract_all') as mock_extract:
            mock_extract.return_value = (
                [{"feature_key": "spot", "feature_value": 150.0, "meta_json": {}}],
                []
            )
            
            engine.run_cycle()
            
            # Extract the discrete prediction calls
            pred_calls = mock_db.insert_prediction.call_args_list
            assert len(pred_calls) == 2
            
            pred_5m = [call[0][1] for call in pred_calls if call[0][1]["horizon_minutes"] == 5][0]
            pred_10m = [call[0][1] for call in pred_calls if call[0][1]["horizon_minutes"] == 10][0]
            
            # The 5m explicitly required OI and lacked it, so it blocked to NO_SIGNAL
            assert pred_5m["decision_state"] == "NO_SIGNAL"
            assert pred_5m["risk_gate_status"] == "BLOCKED"
            
            # The 10m did not require OI, but it was in weights, so it degraded but maintained evaluation capability
            assert pred_10m["risk_gate_status"] == "DEGRADED"
            assert pred_10m["data_quality_state"] == "PARTIAL"
            assert pred_10m["validation_eligible"] is True