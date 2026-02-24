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

def test_integration_degraded_dataset_emits_no_signal(caplog):
    """
    EVIDENCE: Full integration/replay run proving that a dataset missing critical
    RTH features (OI, flow) emits a NO_SIGNAL bounded prediction with explicit
    gate enumeration and triggers the required operational counters.
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

        # Open market (RTH mapping requires spot, net_gex_sign, smart_whale_pressure, oi_pressure)
        mock_mh = MagicMock()
        mock_mh.is_trading_day = True
        mock_mh.ingest_start_et = dt.datetime.now() - dt.timedelta(hours=1)
        mock_mh.ingest_end_et = dt.datetime.now() + dt.timedelta(hours=1)
        mock_mh.get_session_label.return_value = "RTH"
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
        
        # We inject only spot into the extract_all pipeline. The RTH criticality checker
        # will immediately notice that net_gex_sign, smart_whale_pressure, and oi_pressure are missing.
        with patch('src.ingest_engine.extract_all') as mock_extract:
            mock_extract.return_value = (
                [{"feature_key": "spot", "feature_value": 150.0, "meta_json": {}}],
                []
            )
            
            with caplog.at_level(logging.WARNING):
                engine.run_cycle()
            
            # Assert operational counters fired 
            assert "critical_feature_missing_count" in caplog.text
            assert "smart_whale_pressure" in caplog.text
            assert "oi_pressure" in caplog.text
            assert "no_signal_due_to_critical_missing_count" in caplog.text
            
            # Verify the blocked NO_SIGNAL outcome persisted
            pred_call = mock_db.insert_prediction.call_args[0][1]
            
            assert pred_call["decision_state"] == "NO_SIGNAL"
            assert pred_call["risk_gate_status"] == "BLOCKED"
            assert pred_call["is_mock"] is True # Maps from `not validation_eligible`
            assert pred_call["validation_eligible"] is False
            
            # Validate gate reason enumeration traceability
            gate_json = pred_call["gate_json"]
            assert "smart_whale_pressure" in gate_json["critical_features_missing"]
            assert "oi_pressure" in gate_json["critical_features_missing"]
            assert any("critical_features_missing" in reason for reason in gate_json["blocked_reasons"])