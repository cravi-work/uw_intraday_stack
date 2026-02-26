import pytest
import datetime as dt
import tempfile
import os
import logging
from unittest.mock import MagicMock, patch
from src.features import extract_all
from src.endpoint_truth import EndpointContext
from src.ingest_engine import IngestionEngine
from src.replay_engine import run_replay

def test_no_hallucination_replay_defaults():
    ctx = EndpointContext(
        endpoint_id=1, method="GET", path="/api/stock/{ticker}/ohlc/{candle_size}",
        operation_id="opt", signature="GET /api/stock/{ticker}/ohlc/1m",
        used_event_id=None, payload_class="ERROR", freshness_state="ERROR",
        stale_age_min=None, na_reason="missing_raw_payload_for_lineage"
    )
    
    effective_payloads = {1: None}
    contexts = {1: ctx}
    
    f_rows, _ = extract_all(effective_payloads, contexts)
    
    spot_row = next((f for f in f_rows if f["feature_key"] == "spot"), None)
    assert spot_row is not None
    assert spot_row["feature_value"] is None
    assert spot_row["meta_json"]["na_reason"] == "missing_raw_payload_for_lineage"

def test_replay_parity_matches_ingest(caplog):
    # Ensure DuckDB initializes gracefully by passing an explicitly unallocated path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as db_file:
        db_path = db_file.name
    os.remove(db_path)
        
    cfg = {
        "ingestion": {
            "watchlist": ["AAPL"], "cadence_minutes": 5, "enable_market_context": False,
            "premarket_start_et": "04:00", "regular_start_et": "09:30", "regular_end_et": "16:00",
            "afterhours_end_et": "20:00", "ingest_start_et": "04:00", "ingest_end_et": "20:00"
        },
        "storage": {"duckdb_path": db_path, "cycle_lock_path": db_path + ".lock", "writer_lock_path": db_path + ".wlock"},
        "system": {},
        "network": {},
        "validation": {
            "use_default_required_features": False,
            "horizons_minutes": [5],
            "horizon_critical_features": {"5": ["spot", "oi_pressure"]},
            "horizon_weights": {"5": {"spot": 1.0, "oi_pressure": 1.0}}
        }
    }
    
    now_utc = dt.datetime.now(dt.timezone.utc)
    future_utc = (now_utc + dt.timedelta(minutes=5)).isoformat()
    past_utc = (now_utc - dt.timedelta(minutes=15)).isoformat()
    
    valid_meta = {"source_endpoints": [], "freshness_state": "FRESH", "stale_age_min": 0, "na_reason": None, "details": {}}
    
    features = [
        {"feature_key": "spot", "feature_value": 150.0, "meta_json": {**valid_meta, "metric_lineage": {"effective_ts_utc": past_utc}}},
        {"feature_key": "oi_pressure", "feature_value": 1000.0, "meta_json": {**valid_meta, "metric_lineage": {"effective_ts_utc": future_utc}}}
    ]
    
    with patch("src.ingest_engine.get_market_hours") as mock_gmh, \
         patch("src.ingest_engine.fetch_all") as mock_fetch, \
         patch("src.ingest_engine.load_endpoint_plan") as mock_lep, \
         patch("src.ingest_engine.load_api_catalog"), \
         patch("src.ingest_engine.validate_plan_coverage"), \
         patch("src.ingest_engine.FileLock"):

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

        engine = IngestionEngine(cfg=cfg, catalog_path="api_catalog.generated.yaml", config_path="src/config/config.yaml")
        
        with patch('src.ingest_engine.extract_all') as mock_extract:
            mock_extract.return_value = (features, [])
            with caplog.at_level(logging.WARNING):
                engine.run_cycle()
                
    try:
        run_replay(db_path, "AAPL", cfg=cfg)
    finally:
        for ext in ["", ".lock", ".wlock"]:
            if os.path.exists(db_path + ext):
                try:
                    os.remove(db_path + ext)
                except OSError:
                    pass