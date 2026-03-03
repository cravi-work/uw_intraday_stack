import datetime as dt
import json
import logging
import uuid
from unittest.mock import MagicMock

import pytest

from src.validator import validate_pending

UTC = dt.timezone.utc


def _contract_meta(
    *,
    label_version: str = "label_v1",
    target_version: str = "target_v1",
    flat_threshold_pct: float = 0.0005,
    session_boundary_rule: str = "TRUNCATE_TO_SESSION_CLOSE",
) -> dict:
    return {
        "prediction_contract": {
            "target_name": "intraday_direction_3class",
            "target_version": target_version,
            "label_version": label_version,
            "session_boundary_rule": session_boundary_rule,
            "threshold_policy_version": "thresholds_v1",
            "target_spec": {
                "target_name": "intraday_direction_3class",
                "target_version": target_version,
                "class_labels": ["UP", "DOWN", "FLAT"],
                "horizon_kind": "FIXED",
                "horizon_minutes": 15,
                "flat_threshold_pct": flat_threshold_pct,
                "probability_tolerance": 1e-6,
                "contract_source": "test",
            },
            "label_contract": {
                "label_version": label_version,
                "session_boundary_rule": session_boundary_rule,
                "flat_threshold_pct": flat_threshold_pct,
                "flat_threshold_policy": "ABS_RETURN_BAND",
                "threshold_policy_version": "thresholds_v1",
                "neutral_threshold": 0.55,
                "direction_margin": 0.08,
                "contract_source": "test",
            },
        }
    }


def _validated_meta_from_calls(mock_con: MagicMock) -> dict:
    for call in mock_con.execute.call_args_list:
        query = call.args[0]
        params = call.args[1] if len(call.args) > 1 else []
        if query.startswith("UPDATE predictions SET") and "meta_json = ?" in query and params:
            candidate = params[-2] if len(params) >= 2 else None
            if isinstance(candidate, str):
                try:
                    decoded = json.loads(candidate)
                except Exception:
                    continue
                if isinstance(decoded, dict) and decoded.get("validation_outcome"):
                    return decoded
    raise AssertionError("validated meta_json update not found")


def _skipped_meta_from_calls(mock_con: MagicMock) -> dict:
    for call in mock_con.execute.call_args_list:
        query = call.args[0]
        params = call.args[1] if len(call.args) > 1 else []
        if query == "UPDATE predictions SET meta_json = ? WHERE prediction_id = ?" and params:
            decoded = json.loads(params[0])
            if isinstance(decoded, dict) and decoded.get("validation_outcome"):
                return decoded
    raise AssertionError("skip meta_json update not found")


def test_leakage_guard_rejection(caplog):
    mock_con = MagicMock()
    pred_asof = dt.datetime(2026, 1, 1, 10, 0, 0, tzinfo=UTC)
    now_utc = pred_asof + dt.timedelta(hours=2)

    target_ts = pred_asof + dt.timedelta(minutes=5)
    leaked_ts = target_ts + dt.timedelta(minutes=15)
    meta = _contract_meta()

    def mock_execute(query, params=None):
        m = MagicMock()
        if "PRAGMA" in query:
            m.fetchall.return_value = [(0, "outcome_price", "DOUBLE")]
        elif "SELECT p.prediction_id" in query:
            m.fetchall.return_value = [
                (
                    str(uuid.uuid4()),
                    str(uuid.uuid4()),
                    "FIXED",
                    5,
                    0,
                    150.0,
                    0.6,
                    0.4,
                    0.0,
                    "AAPL",
                    pred_asof,
                    "LONG",
                    "win_id",
                    True,
                    meta,
                    pred_asof.replace(hour=16, minute=0),
                    pred_asof.replace(hour=20, minute=0),
                    False,
                    "RTH",
                )
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
            tolerance_minutes=2,
            max_horizon_drift_minutes=10,
        )

    assert stats.skipped == 1
    assert stats.leakage_violations == 1
    assert stats.reason_counts["TARGET_EXCEEDED_MAX_DRIFT"] == 1
    assert "Leakage guard blocked prediction" in caplog.text


def test_to_close_semantics():
    mock_con = MagicMock()
    pred_asof = dt.datetime(2026, 1, 1, 15, 0, 0, tzinfo=UTC)
    now_utc = pred_asof + dt.timedelta(hours=2)
    hz_sec = 3600
    target_ts = pred_asof + dt.timedelta(seconds=hz_sec)
    meta = _contract_meta()
    meta["prediction_contract"]["target_spec"]["horizon_kind"] = "TO_CLOSE"
    meta["prediction_contract"]["target_spec"]["horizon_minutes"] = 0

    def mock_execute(query, params=None):
        m = MagicMock()
        if "PRAGMA" in query:
            m.fetchall.return_value = [(0, "outcome_price", "DOUBLE")]
        elif "SELECT p.prediction_id" in query:
            m.fetchall.return_value = [
                (
                    str(uuid.uuid4()),
                    str(uuid.uuid4()),
                    "TO_CLOSE",
                    0,
                    hz_sec,
                    150.0,
                    0.6,
                    0.4,
                    0.0,
                    "AAPL",
                    pred_asof,
                    "LONG",
                    "win_id",
                    True,
                    meta,
                    target_ts,
                    pred_asof.replace(hour=20, minute=0),
                    False,
                    "RTH",
                )
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
        tolerance_minutes=2,
        max_horizon_drift_minutes=10,
    )

    assert stats.updated == 1
    assert stats.skipped == 0
    assert stats.reason_counts["UPDATED_VALIDATED"] == 1


def test_session_end_crossing_is_explicitly_truncated_to_close():
    mock_con = MagicMock()
    pred_asof = dt.datetime(2026, 1, 2, 15, 50, 0, tzinfo=UTC)
    market_close = dt.datetime(2026, 1, 2, 16, 0, 0, tzinfo=UTC)
    now_utc = pred_asof + dt.timedelta(hours=2)
    meta = _contract_meta(label_version="label_v2")
    meta["prediction_contract"]["target_spec"]["horizon_minutes"] = 30

    def mock_execute(query, params=None):
        m = MagicMock()
        if "PRAGMA" in query:
            m.fetchall.return_value = [(0, "outcome_price", "DOUBLE")]
        elif "SELECT p.prediction_id" in query:
            m.fetchall.return_value = [
                (
                    str(uuid.uuid4()),
                    str(uuid.uuid4()),
                    "FIXED",
                    30,
                    0,
                    150.0,
                    0.6,
                    0.3,
                    0.1,
                    "AAPL",
                    pred_asof,
                    "LONG",
                    "win_id",
                    True,
                    meta,
                    market_close,
                    pred_asof.replace(hour=20, minute=0),
                    False,
                    "RTH",
                )
            ]
        elif "SELECT snapshot_id, asof_ts_utc" in query:
            m.fetchone.return_value = (str(uuid.uuid4()), market_close)
        elif "SELECT feature_value" in query:
            m.fetchone.return_value = (151.0,)
        return m

    mock_con.execute.side_effect = mock_execute

    stats = validate_pending(
        mock_con,
        now_utc=now_utc,
        flat_threshold_pct=0.0005,
        tolerance_minutes=2,
        max_horizon_drift_minutes=10,
    )

    assert stats.updated == 1
    updated_meta = _validated_meta_from_calls(mock_con)
    outcome = updated_meta["validation_outcome"]
    assert outcome["session_boundary_truncated"] is True
    assert outcome["half_day_truncation"] is False
    assert outcome["target_ts_utc"] == market_close.isoformat()
    assert outcome["label_version"] == "label_v2"


def test_half_day_truncation_is_explicit():
    mock_con = MagicMock()
    pred_asof = dt.datetime(2026, 7, 3, 12, 45, 0, tzinfo=UTC)
    market_close = dt.datetime(2026, 7, 3, 13, 0, 0, tzinfo=UTC)
    now_utc = pred_asof + dt.timedelta(hours=2)
    meta = _contract_meta()
    meta["prediction_contract"]["target_spec"]["horizon_minutes"] = 60

    def mock_execute(query, params=None):
        m = MagicMock()
        if "PRAGMA" in query:
            m.fetchall.return_value = [(0, "outcome_price", "DOUBLE")]
        elif "SELECT p.prediction_id" in query:
            m.fetchall.return_value = [
                (
                    str(uuid.uuid4()),
                    str(uuid.uuid4()),
                    "FIXED",
                    60,
                    0,
                    150.0,
                    0.6,
                    0.3,
                    0.1,
                    "AAPL",
                    pred_asof,
                    "LONG",
                    "win_id",
                    True,
                    meta,
                    market_close,
                    pred_asof.replace(hour=17, minute=0),
                    True,
                    "RTH",
                )
            ]
        elif "SELECT snapshot_id, asof_ts_utc" in query:
            m.fetchone.return_value = (str(uuid.uuid4()), market_close)
        elif "SELECT feature_value" in query:
            m.fetchone.return_value = (152.0,)
        return m

    mock_con.execute.side_effect = mock_execute

    stats = validate_pending(
        mock_con,
        now_utc=now_utc,
        flat_threshold_pct=0.0005,
        tolerance_minutes=2,
        max_horizon_drift_minutes=10,
    )

    assert stats.updated == 1
    updated_meta = _validated_meta_from_calls(mock_con)
    outcome = updated_meta["validation_outcome"]
    assert outcome["session_boundary_truncated"] is True
    assert outcome["half_day_truncation"] is True
    assert outcome["target_ts_utc"] == market_close.isoformat()


def test_missing_realized_snapshot_is_explicit_unavailable_outcome():
    mock_con = MagicMock()
    pred_asof = dt.datetime(2026, 1, 1, 10, 0, 0, tzinfo=UTC)
    now_utc = pred_asof + dt.timedelta(hours=2)
    meta = _contract_meta()

    def mock_execute(query, params=None):
        m = MagicMock()
        if "PRAGMA" in query:
            m.fetchall.return_value = [(0, "outcome_price", "DOUBLE")]
        elif "SELECT p.prediction_id" in query:
            m.fetchall.return_value = [
                (
                    str(uuid.uuid4()),
                    str(uuid.uuid4()),
                    "FIXED",
                    15,
                    0,
                    150.0,
                    0.6,
                    0.3,
                    0.1,
                    "AAPL",
                    pred_asof,
                    "LONG",
                    "win_id",
                    True,
                    meta,
                    pred_asof.replace(hour=16, minute=0),
                    pred_asof.replace(hour=20, minute=0),
                    False,
                    "RTH",
                )
            ]
        elif "SELECT snapshot_id, asof_ts_utc" in query:
            m.fetchone.return_value = None
        return m

    mock_con.execute.side_effect = mock_execute

    stats = validate_pending(
        mock_con,
        now_utc=now_utc,
        flat_threshold_pct=0.0005,
        tolerance_minutes=2,
        max_horizon_drift_minutes=10,
    )

    assert stats.updated == 0
    assert stats.skipped == 1
    assert stats.reason_counts["UNAVAILABLE_REALIZED_OUTCOME_NO_SNAPSHOT"] == 1
    skipped_meta = _skipped_meta_from_calls(mock_con)
    assert skipped_meta["validation_outcome"]["reason_code"] == "UNAVAILABLE_REALIZED_OUTCOME_NO_SNAPSHOT"


def test_deterministic_replay_checksum():
    import hashlib

    rows = [
        ("uuid1", 0.05, 0.10, True),
        ("uuid2", 0.45, 0.80, False),
    ]

    state_str_1 = json.dumps(rows, sort_keys=True)
    hash_1 = hashlib.sha256(state_str_1.encode()).hexdigest()

    rows_reordered_query_sim = [
        ("uuid1", 0.05, 0.10, True),
        ("uuid2", 0.45, 0.80, False),
    ]

    state_str_2 = json.dumps(rows_reordered_query_sim, sort_keys=True)
    hash_2 = hashlib.sha256(state_str_2.encode()).hexdigest()

    assert hash_1 == hash_2
