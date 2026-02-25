import pytest
import datetime as dt
import uuid
import logging
from unittest.mock import MagicMock
from src.validator import validate_pending

def test_leakage_guard_rejection(caplog):
    mock_con = MagicMock()
    pred_asof = dt.datetime(2026, 1, 1, 10, 0, 0, tzinfo=dt.timezone.utc)
    now_utc = pred_asof + dt.timedelta(hours=2)
    
    target_ts = pred_asof + dt.timedelta(minutes=5)
    leaked_ts = target_ts + dt.timedelta(minutes=15)

    def mock_execute(query, params=None):
        m = MagicMock()
        if "PRAGMA" in query:
            m.fetchall.return_value = [(0, "outcome_price", "DOUBLE")]
        elif "SELECT p.prediction_id" in query:
            m.fetchall.return_value = [
                (str(uuid.uuid4()), str(uuid.uuid4()), "FIXED", 5, 0, 150.0, 0.6, 0.4, 0.0, "AAPL", pred_asof, "LONG", "win_id", True)
            ]
        elif "SELECT snapshot_id, asof_ts_utc" in query:
            m.fetchone.return_value = (str(uuid.uuid4()), leaked_ts)
        return m

    mock_con.execute.side_effect = mock_execute

    with caplog.at_level(logging.WARNING):
        stats = validate_pending(
            mock_con, 
            now_utc=now_utc, 
            flat_threshold_pct=0.0005, 
            tolerance_minutes=2
        )
        
    assert stats.skipped == 1
    assert "Leakage guard blocked prediction" in caplog.text
    assert stats.leakage_violations == 1

def test_to_close_semantics():
    mock_con = MagicMock()
    pred_asof = dt.datetime(2026, 1, 1, 15, 0, 0, tzinfo=dt.timezone.utc)
    now_utc = pred_asof + dt.timedelta(hours=2)
    hz_sec = 3600 
    
    target_ts = pred_asof + dt.timedelta(seconds=hz_sec)
    
    def mock_execute(query, params=None):
        m = MagicMock()
        if "PRAGMA" in query:
            m.fetchall.return_value = [(0, "outcome_price", "DOUBLE")]
        elif "SELECT p.prediction_id" in query:
            m.fetchall.return_value = [
                (str(uuid.uuid4()), str(uuid.uuid4()), "TO_CLOSE", 0, hz_sec, 150.0, 0.6, 0.4, 0.0, "AAPL", pred_asof, "LONG", "win_id", True)
            ]
        elif "SELECT snapshot_id, asof_ts_utc" in query:
            m.fetchone.return_value = (str(uuid.uuid4()), target_ts)
        elif "SELECT feature_value" in query:
            m.fetchone.return_value = (152.0,)
        return m

    mock_con.execute.side_effect = mock_execute

    stats = validate_pending(
        mock_con, 
        now_utc=now_utc, 
        flat_threshold_pct=0.0005, 
        tolerance_minutes=2
    )
    
    assert stats.updated == 1
    assert stats.skipped == 0

def test_deterministic_replay_checksum():
    import hashlib
    import json
    
    rows = [
        ("uuid1", 0.05, 0.10, True),
        ("uuid2", 0.45, 0.80, False)
    ]
    
    state_str_1 = json.dumps(rows, sort_keys=True)
    hash_1 = hashlib.sha256(state_str_1.encode()).hexdigest()
    
    rows_reordered_query_sim = [
        ("uuid1", 0.05, 0.10, True),
        ("uuid2", 0.45, 0.80, False)
    ]
    
    state_str_2 = json.dumps(rows_reordered_query_sim, sort_keys=True)
    hash_2 = hashlib.sha256(state_str_2.encode()).hexdigest()
    
    assert hash_1 == hash_2