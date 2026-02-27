# tests/test_horizon_target_gate.py
import pytest
import datetime as dt
import logging
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
            "use_default_required_features": False,
            "emit_to_close_horizon": True,
            "horizon_weights_source": "explicit",
            "horizons_minutes": [15, 60],
            "horizon_critical_features": {
                "15": [], 
                "60": ["spot"], 
                "to_close": []
            },
            "horizon_weights": {
                "15": {"oi_pressure": 1.0},
                "60": {"spot": 1.0, "oi_pressure": 0.5},
                "to_close": {"oi_pressure": 1.0}
            }
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

def test_zero_target_features_yields_no_signal(mock_engine_env, caplog):
    fixed_asof = dt.datetime(2026, 1, 1, 12, 0, tzinfo=dt.timezone.utc)
    valid_meta = {"source_endpoints": [], "freshness_state": "FRESH", "stale_age_min": 0, "na_reason": None, "details": {}}
    past_utc = (fixed_asof - dt.timedelta(minutes=15)).isoformat()
    
    # We explicitly pass ONLY a non-target feature. 
    # The actual explicitly contracted targets (oi_pressure, spot) are missing.
    features = [
        {"feature_key": "some_other_feature", "feature_value": 150.0, "meta_json": {**valid_meta, "metric_lineage": {"effective_ts_utc": past_utc}}},
    ]
    
    mock_db = _run_with_features(mock_engine_env, features, caplog, fixed_asof)
    
    calls = [call[0][1] for call in mock_db.insert_prediction.call_args_list]
    
    # Assert Horizon 15: It has NO criticals, but also zero valid targets. It MUST hard-block.
    pred_15 = next(c for c in calls if c["horizon_kind"] == "FIXED" and c["horizon_minutes"] == 15)
    assert pred_15["decision_state"] == "NO_SIGNAL"
    assert pred_15["validation_eligible"] is False
    assert "no_horizon_target_features_after_alignment" in pred_15["blocked_reasons"]
    
    # Assert Horizon 60: Has criticals + zero targets. It MUST hard-block.
    pred_60 = next(c for c in calls if c["horizon_kind"] == "FIXED" and c["horizon_minutes"] == 60)
    assert pred_60["decision_state"] == "NO_SIGNAL"
    assert pred_60["validation_eligible"] is False
    assert "no_horizon_target_features_after_alignment" in pred_60["blocked_reasons"]
    
    # Assert TO_CLOSE horizon: Must also hard-block.
    pred_tc = next(c for c in calls if c["horizon_kind"] == "TO_CLOSE")
    assert pred_tc["decision_state"] == "NO_SIGNAL"
    assert pred_tc["validation_eligible"] is False
    assert "no_horizon_target_features_after_alignment" in pred_tc["blocked_reasons"]