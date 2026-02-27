# tests/test_replay_parity_contract.py
import pytest
import datetime as dt
import tempfile
import os
import json
import uuid
import duckdb
from src.replay_engine import run_replay

@pytest.fixture
def test_db_path():
    # Generate a unique path but DO NOT create an empty file. 
    # Let DuckDB create and format it directly to prevent 0-byte header errors.
    path = os.path.join(tempfile.gettempdir(), f"test_parity_{uuid.uuid4().hex}.duckdb")
    
    con = duckdb.connect(path)
    con.execute("""
        CREATE TABLE snapshots (snapshot_id VARCHAR, asof_ts_utc TIMESTAMPTZ, ticker VARCHAR, session_label VARCHAR, is_trading_day BOOLEAN, is_early_close BOOLEAN, data_quality_score DOUBLE, market_close_utc TIMESTAMPTZ, post_end_utc TIMESTAMPTZ, seconds_to_close DOUBLE);
        CREATE TABLE snapshot_lineage (snapshot_id VARCHAR, endpoint_id INTEGER, used_event_id VARCHAR, freshness_state VARCHAR, data_age_seconds DOUBLE, payload_class VARCHAR, na_reason VARCHAR, meta_json JSON);
        CREATE TABLE raw_http_events (event_id VARCHAR, run_id VARCHAR, ticker VARCHAR, endpoint_id INTEGER, requested_at_utc DOUBLE, received_at_utc DOUBLE, status_code INTEGER, latency_ms DOUBLE, payload_hash VARCHAR, payload_json JSON, is_retry BOOLEAN, error_type VARCHAR, error_message VARCHAR, circuit_breaker_state VARCHAR);
        CREATE TABLE dim_endpoints (endpoint_id INTEGER, method VARCHAR, path VARCHAR, query_params JSON, signature VARCHAR);
        CREATE TABLE predictions (prediction_id VARCHAR, snapshot_id VARCHAR, horizon_minutes INTEGER, horizon_kind VARCHAR, horizon_seconds INTEGER, start_price DOUBLE, bias DOUBLE, confidence DOUBLE, prob_up DOUBLE, prob_down DOUBLE, prob_flat DOUBLE, model_name VARCHAR, model_version VARCHAR, model_hash VARCHAR, is_mock BOOLEAN, meta_json JSON, decision_state VARCHAR, risk_gate_status VARCHAR, confidence_state VARCHAR, data_quality_state VARCHAR, blocked_reasons JSON, degraded_reasons JSON, validation_eligible BOOLEAN, gate_json JSON, source_ts_min_utc TIMESTAMPTZ, source_ts_max_utc TIMESTAMPTZ, critical_missing_count INTEGER, alignment_status VARCHAR, decision_window_id VARCHAR);
    """)
    con.close()
    
    yield path
    
    # Safe cleanup across OS environments
    for ext in ["", ".wal", ".tmp"]:
        p = path + ext
        if os.path.exists(p):
            try:
                os.remove(p)
            except Exception:
                pass

def test_replay_missing_lineage_field_fails_parity(test_db_path):
    con = duckdb.connect(test_db_path)
    now = dt.datetime(2026, 1, 1, 12, 0, tzinfo=dt.timezone.utc)
    
    con.execute("INSERT INTO snapshots VALUES ('s1', ?, 'AAPL', 'RTH', true, false, 1.0, ?, ?, 3600)", [now, now, now])
    
    # FIX: Use a real registered extractor path to bypass the CRITICAL COVERAGE GAP safety trap
    con.execute("INSERT INTO dim_endpoints VALUES (1, 'GET', '/api/stock/{ticker}/ohlc/{candle_size}', '{}', 'GET /api/stock/{ticker}/ohlc/1m')")
    
    con.execute("INSERT INTO raw_http_events VALUES ('e1', 'r1', 'AAPL', 1, 0.0, 0.1, 200, 100, 'hash', '[]', false, NULL, NULL, 'CLOSED')")
    
    bad_meta = {
        "source_endpoints": [], "freshness_state": "FRESH", "stale_age_min": 0, "na_reason": None,
        "details": {
            # "effective_ts_utc" IS DELIBERATELY MISSING
            "endpoint_asof_ts_utc": now.isoformat(),
            "truth_status": "SUCCESS_HAS_DATA",
            "stale_age_seconds": 0
        }
    }
    
    con.execute("INSERT INTO snapshot_lineage VALUES ('s1', 1, 'e1', 'FRESH', 0, 'SUCCESS_HAS_DATA', NULL, ?)", [json.dumps(bad_meta)])
    
    # We assert that historically, this was 'PASS' and 'LONG'.
    # Because of the missing lineage, replay will downgrade it and fail parity.
    con.execute("INSERT INTO predictions VALUES ('p1', 's1', 15, 'FIXED', NULL, 150.0, 0.5, 0.5, 0.5, 0.3, 0.2, 'model', '1', 'hash', false, '{}', 'LONG', 'PASS', 'MEDIUM', 'VALID', '[]', '[]', true, '{}', ?, ?, 0, 'ALIGNED', 'w1')", [now, now])
    con.close()

    cfg = {
        "ingestion": {"watchlist": ["AAPL"], "cadence_minutes": 5},
        "validation": {
            "alignment_tolerance_sec": 900, "use_default_required_features": False, "emit_to_close_horizon": False,
            "horizon_weights_source": "explicit", "horizons_minutes": [15],
            "horizon_critical_features": {"15": []},
            "horizon_weights": {"15": {"spot": 1.0}}
        }
    }
    
    with pytest.raises(RuntimeError, match="PARITY MISMATCH.*Risk/Decision Governance altered"):
        run_replay(test_db_path, "AAPL", cfg=cfg)

def test_replay_cannot_compute_silent_skip(test_db_path):
    con = duckdb.connect(test_db_path)
    now = dt.datetime(2026, 1, 1, 12, 0, tzinfo=dt.timezone.utc)
    con.execute("INSERT INTO snapshots VALUES ('s1', ?, 'AAPL', 'RTH', true, false, 1.0, ?, ?, 3600)", [now, now, now])
    
    # Insert a prediction for horizon 60
    con.execute("INSERT INTO predictions VALUES ('p1', 's1', 60, 'FIXED', NULL, 150.0, 0.5, 0.5, 0.5, 0.3, 0.2, 'model', '1', 'hash', false, '{}', 'LONG', 'PASS', 'MEDIUM', 'VALID', '[]', '[]', true, '{}', ?, ?, 0, 'ALIGNED', 'w1')", [now, now])
    con.close()
    
    cfg = {
        "ingestion": {"watchlist": ["AAPL"], "cadence_minutes": 5},
        "validation": {
            "alignment_tolerance_sec": 900, "use_default_required_features": False, "emit_to_close_horizon": False,
            "horizon_weights_source": "explicit",
            # Config only has 15, so replay cannot compute 60! This must throw an explicit error.
            "horizons_minutes": [15], 
            "horizon_critical_features": {"15": []},
            "horizon_weights": {"15": {"spot": 1.0}}
        }
    }
    
    with pytest.raises(RuntimeError, match="exists in stored but replay could not compute it"):
        run_replay(test_db_path, "AAPL", cfg=cfg)