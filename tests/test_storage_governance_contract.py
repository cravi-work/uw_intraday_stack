import json
from datetime import datetime, timezone

import duckdb

from src.storage import DbWriter

UTC = timezone.utc


def _parse_jsonish(value):
    if value is None:
        return None
    if isinstance(value, str):
        return json.loads(value)
    return value


def _bootstrap_snapshot(db: DbWriter):
    now = datetime(2026, 2, 3, 15, 30, tzinfo=UTC)
    with db.writer() as con:
        db.ensure_schema(con)
        cfg_ver = db.insert_config(con, "model: {}\n")
        run_id = db.begin_run(
            con,
            asof_ts_utc=now,
            session_label="RTH",
            is_trading_day=True,
            is_early_close=False,
            config_version=cfg_ver,
            api_catalog_hash="catalog_hash_v2",
        )
        snapshot_id = db.insert_snapshot(
            con,
            run_id=run_id,
            asof_ts_utc=now,
            ticker="AAPL",
            session_label="RTH",
            is_trading_day=True,
            is_early_close=False,
            data_quality_score=1.0,
            market_close_utc=now,
            post_end_utc=now,
            seconds_to_close=1800,
        )
    return snapshot_id, now


def _governed_scope(session: str = "RTH"):
    return {
        "horizon_kind": "FIXED",
        "horizon_minutes": 15,
        "session": session,
        "regime": "DEFAULT",
        "replay_mode": "UNKNOWN",
        "scope_contract_version": "calibration_scope/v1",
    }


def _governed_meta(*, session: str = "RTH", ood_reason: str = "time_provenance_degraded:spot"):
    scope = _governed_scope(session=session)
    return {
        "prediction_contract": {
            "target_name": "intraday_direction_3class",
            "target_version": "target_v2",
            "label_version": "label_v4",
            "threshold_policy_version": "threshold_v5",
            "calibration_scope": scope,
            "target_spec": {
                "target_name": "intraday_direction_3class",
                "target_version": "target_v2",
            },
            "label_contract": {
                "label_version": "label_v4",
                "threshold_policy_version": "threshold_v5",
            },
        },
        "probability_contract": {
            "target_spec": {
                "target_name": "intraday_direction_3class",
                "target_version": "target_v2",
            },
            "calibration_artifact_ref": {
                "artifact_name": "phase1_calibration",
                "artifact_version": "cal_v9",
                "target_name": "intraday_direction_3class",
                "target_version": "target_v2",
                "artifact_hash": "artifact_hash_v9",
                "calibration_scope": scope,
            },
            "ood_state": "DEGRADED",
            "ood_reason": ood_reason,
            "suppression_reason": None,
        },
        "calibration_selection": {
            "reason_code": "selected",
            "artifact_hash": "artifact_hash_v9",
            "calibration_scope": scope,
        },
        "ood_state": "DEGRADED",
        "ood_reason": ood_reason,
        "ood_assessment": {
            "primary_reason": ood_reason,
            "state": "DEGRADED",
        },
        "horizon_contract": {
            "feature_contracts": {
                "spot": {"contract_version": "decision_path/v1"},
                "oi_pressure": {"contract_version": "decision_path/v1"},
            }
        },
        "replay_mode": "UNKNOWN",
    }


def _prediction_payload(snapshot_id: str, *, confidence: float = 0.61, meta_json=None):
    return {
        "snapshot_id": snapshot_id,
        "horizon_minutes": 15,
        "horizon_kind": "FIXED",
        "horizon_seconds": None,
        "start_price": 188.25,
        "bias": 0.18,
        "confidence": confidence,
        "prob_up": 0.56,
        "prob_down": 0.18,
        "prob_flat": 0.26,
        "model_name": "phase1_additive",
        "model_version": "model_v9",
        "model_hash": "model_hash_v9",
        "is_mock": False,
        "meta_json": meta_json or _governed_meta(),
        "decision_state": "LONG",
        "risk_gate_status": "PASS",
        "data_quality_state": "DEGRADED",
        "confidence_state": "DEGRADED",
        "blocked_reasons": [],
        "degraded_reasons": ["time_provenance_degraded:spot"],
        "validation_eligible": True,
        "gate_json": {"validation_eligible": True},
        "alignment_status": "ALIGNED",
        "source_ts_min_utc": datetime(2026, 2, 3, 15, 29, tzinfo=UTC),
        "source_ts_max_utc": datetime(2026, 2, 3, 15, 30, tzinfo=UTC),
        "critical_missing_count": 0,
        "decision_window_id": "fixed_15m_window",
    }


def test_storage_schema_includes_governance_columns(tmp_path):
    db = DbWriter(str(tmp_path / "governance_schema.duckdb"), str(tmp_path / "governance_schema.lock"))
    with db.writer() as con:
        db.ensure_schema(con)
        pred_cols = {row[1] for row in con.execute("PRAGMA table_info('predictions')").fetchall()}
        trace_cols = {row[1] for row in con.execute("PRAGMA table_info('decision_traces')").fetchall()}

    expected = {
        "ood_reason",
        "calibration_scope",
        "calibration_artifact_hash",
        "decision_path_contract_version",
    }
    assert expected.issubset(pred_cols)
    assert expected.issubset(trace_cols)


def test_prediction_and_decision_trace_persist_governance_state_idempotently(tmp_path):
    db = DbWriter(str(tmp_path / "governance_insert.duckdb"), str(tmp_path / "governance_insert.lock"))
    snapshot_id, _ = _bootstrap_snapshot(db)
    payload = _prediction_payload(snapshot_id)

    with db.writer() as con:
        first_id = db.insert_prediction(con, payload)
        second_id = db.insert_prediction(con, _prediction_payload(snapshot_id, confidence=0.72))
        pred_rows = con.execute(
            """
            SELECT prediction_id, confidence, ood_reason, calibration_scope,
                   calibration_artifact_hash, decision_path_contract_version
            FROM predictions
            """
        ).fetchall()
        trace_rows = con.execute(
            """
            SELECT prediction_id, event_type, ood_reason, calibration_scope,
                   calibration_artifact_hash, decision_path_contract_version
            FROM decision_traces
            """
        ).fetchall()

    assert first_id == second_id
    assert len(pred_rows) == 1
    assert len(trace_rows) == 1

    pred_row = pred_rows[0]
    pred_scope = _parse_jsonish(pred_row[3])
    assert str(pred_row[0]) == first_id
    assert pred_row[1] == 0.72
    assert pred_row[2] == "time_provenance_degraded:spot"
    assert pred_scope["session"] == "RTH"
    assert pred_row[4] == "artifact_hash_v9"
    assert pred_row[5] == "decision_path/v1"

    trace_row = trace_rows[0]
    trace_scope = _parse_jsonish(trace_row[3])
    assert str(trace_row[0]) == first_id
    assert trace_row[1] == "signal_degraded"
    assert trace_row[2] == "time_provenance_degraded:spot"
    assert trace_scope["horizon_minutes"] == 15
    assert trace_row[4] == "artifact_hash_v9"
    assert trace_row[5] == "decision_path/v1"


def test_storage_migration_backfills_governance_columns(tmp_path):
    db_path = tmp_path / "governance_migrate.duckdb"
    con = duckdb.connect(str(db_path))
    con.execute(
        """
        CREATE TABLE predictions (
            prediction_id VARCHAR PRIMARY KEY,
            prediction_business_key TEXT,
            snapshot_id VARCHAR,
            horizon_minutes INTEGER,
            horizon_kind TEXT DEFAULT 'FIXED',
            horizon_seconds INTEGER,
            start_price DOUBLE,
            bias DOUBLE,
            confidence DOUBLE,
            prob_up DOUBLE,
            prob_down DOUBLE,
            prob_flat DOUBLE,
            model_name TEXT,
            model_version TEXT,
            model_hash TEXT,
            is_mock BOOLEAN DEFAULT FALSE,
            outcome_realized BOOLEAN DEFAULT FALSE,
            realized_at_utc TIMESTAMP,
            outcome_price DOUBLE,
            outcome_label TEXT,
            brier_score DOUBLE,
            log_loss DOUBLE,
            is_correct BOOLEAN,
            meta_json JSON,
            decision_state TEXT DEFAULT 'UNKNOWN',
            risk_gate_status TEXT DEFAULT 'UNKNOWN',
            data_quality_state TEXT DEFAULT 'UNKNOWN',
            confidence_state TEXT DEFAULT 'UNKNOWN',
            blocked_reasons_json JSON,
            degraded_reasons_json JSON,
            validation_eligible BOOLEAN DEFAULT TRUE,
            gate_json JSON,
            alignment_status TEXT DEFAULT 'UNKNOWN',
            source_ts_min_utc TIMESTAMP,
            source_ts_max_utc TIMESTAMP,
            critical_missing_count INTEGER DEFAULT 0,
            decision_window_id TEXT DEFAULT 'UNKNOWN'
        )
        """
    )
    con.execute(
        """
        CREATE TABLE decision_traces (
            trace_id VARCHAR PRIMARY KEY,
            created_at_utc TIMESTAMP,
            prediction_id VARCHAR,
            prediction_business_key TEXT,
            snapshot_id VARCHAR,
            event_type TEXT,
            decision_state TEXT,
            risk_gate_status TEXT,
            data_quality_state TEXT,
            confidence_state TEXT,
            suppression_reason TEXT,
            ood_state TEXT,
            replay_mode TEXT,
            model_name TEXT,
            model_version TEXT,
            target_name TEXT,
            target_version TEXT,
            calibration_version TEXT,
            threshold_policy_version TEXT,
            blocked_reasons_json JSON,
            degraded_reasons_json JSON,
            trace_json JSON
        )
        """
    )

    legacy_business_key = json.dumps(
        {
            "snapshot_id": "snap-legacy",
            "horizon_kind": "FIXED",
            "horizon_minutes": 15,
            "horizon_seconds": None,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    legacy_meta = _governed_meta(session="PREMARKET", ood_reason="legacy_session_mismatch")
    con.execute(
        """
        INSERT INTO predictions VALUES (
            'pred-legacy', ?, 'snap-legacy', 15, 'FIXED', NULL,
            100.0, 0.2, 0.5, 0.6, 0.2, 0.2,
            'model', 'legacy_model_v2', 'hash1', false,
            false, NULL, NULL, NULL, NULL, NULL, NULL,
            ?, 'LONG', 'PASS', 'DEGRADED', 'DEGRADED', '[]', '[]', true,
            '{}', 'ALIGNED', NULL, NULL, 0, 'legacy_window'
        )
        """,
        [legacy_business_key, json.dumps(legacy_meta)],
    )
    con.execute(
        """
        INSERT INTO decision_traces VALUES (
            'trace-legacy', ?, 'pred-legacy', ?, 'snap-legacy', 'signal_degraded',
            'LONG', 'PASS', 'DEGRADED', 'DEGRADED', NULL, 'DEGRADED', 'UNKNOWN',
            'model', 'legacy_model_v2', 'intraday_direction_3class', 'target_v2',
            'cal_v9', 'threshold_v5', '[]', '[]', '{}'
        )
        """,
        [datetime(2026, 2, 3, 15, 31, tzinfo=UTC), legacy_business_key],
    )
    con.close()

    db = DbWriter(str(db_path), str(tmp_path / "governance_migrate.lock"))
    with db.writer() as con2:
        db.ensure_schema(con2)
        pred_row = con2.execute(
            """
            SELECT ood_reason, calibration_scope, calibration_artifact_hash,
                   decision_path_contract_version
            FROM predictions WHERE prediction_id = 'pred-legacy'
            """
        ).fetchone()
        trace_row = con2.execute(
            """
            SELECT ood_reason, calibration_scope, calibration_artifact_hash,
                   decision_path_contract_version
            FROM decision_traces WHERE trace_id = 'trace-legacy'
            """
        ).fetchone()

    pred_scope = _parse_jsonish(pred_row[1])
    trace_scope = _parse_jsonish(trace_row[1])
    assert pred_row[0] == "legacy_session_mismatch"
    assert pred_scope["session"] == "PREMARKET"
    assert pred_row[2] == "artifact_hash_v9"
    assert pred_row[3] == "decision_path/v1"

    assert trace_row[0] == "legacy_session_mismatch"
    assert trace_scope["session"] == "PREMARKET"
    assert trace_row[2] == "artifact_hash_v9"
    assert trace_row[3] == "decision_path/v1"
