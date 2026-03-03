import json
import uuid
from datetime import datetime, timezone

import duckdb

from src.storage import DbWriter

UTC = timezone.utc


def _bootstrap_snapshot(db: DbWriter):
    now = datetime(2026, 1, 2, 15, 30, tzinfo=UTC)
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
            api_catalog_hash="catalog_hash_v1",
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


def _prediction_meta():
    return {
        "prediction_contract": {
            "target_name": "intraday_direction_3class",
            "target_version": "target_v1",
            "label_version": "label_v2",
            "threshold_policy_version": "threshold_v3",
            "target_spec": {
                "target_name": "intraday_direction_3class",
                "target_version": "target_v1",
            },
            "label_contract": {
                "label_version": "label_v2",
                "threshold_policy_version": "threshold_v3",
            },
        },
        "probability_contract": {
            "target_spec": {
                "target_name": "intraday_direction_3class",
                "target_version": "target_v1",
            },
            "calibration_artifact_ref": {
                "artifact_name": "phase0_calibration",
                "artifact_version": "cal_v7",
                "target_name": "intraday_direction_3class",
                "target_version": "target_v1",
            },
            "ood_state": "UNKNOWN",
            "suppression_reason": None,
        },
        "freshness_registry_diagnostics": {
            "feature_policies": {
                "spot": {
                    "policy": "REQUIRED_LIVE",
                    "lag_class": "LIVE",
                    "join_skew_tolerance_seconds": 60,
                    "max_tolerated_age_seconds": 60,
                    "policy_source": "registry",
                }
            }
        },
        "replay_mode": "UNKNOWN",
        "ood_state": "UNKNOWN",
    }


def _prediction_payload(snapshot_id: str, *, confidence: float = 0.55, prob_up: float = 0.6, model_hash: str = "hash_v1"):
    return {
        "snapshot_id": snapshot_id,
        "horizon_minutes": 15,
        "horizon_kind": "FIXED",
        "horizon_seconds": None,
        "start_price": 100.0,
        "bias": 0.25,
        "confidence": confidence,
        "prob_up": prob_up,
        "prob_down": 0.2,
        "prob_flat": 0.2,
        "model_name": "phase0_additive",
        "model_version": "model_v5",
        "model_hash": model_hash,
        "is_mock": False,
        "meta_json": _prediction_meta(),
        "decision_state": "LONG",
        "risk_gate_status": "PASS",
        "data_quality_state": "VALID",
        "confidence_state": "MEDIUM",
        "blocked_reasons": [],
        "degraded_reasons": [],
        "validation_eligible": True,
        "gate_json": {"validation_eligible": True},
        "alignment_status": "ALIGNED",
        "source_ts_min_utc": datetime(2026, 1, 2, 15, 29, tzinfo=UTC),
        "source_ts_max_utc": datetime(2026, 1, 2, 15, 30, tzinfo=UTC),
        "critical_missing_count": 0,
        "decision_window_id": "window_15m",
    }


def test_storage_schema_contract_columns_present(tmp_path):
    db = DbWriter(str(tmp_path / "contract.duckdb"), str(tmp_path / "contract.lock"))
    with db.writer() as con:
        db.ensure_schema(con)
        pred_cols = {row[1] for row in con.execute("PRAGMA table_info('predictions')").fetchall()}
        raw_cols = {row[1] for row in con.execute("PRAGMA table_info('raw_http_events')").fetchall()}

    assert {
        "prediction_business_key",
        "target_name",
        "target_version",
        "label_version",
        "feature_version",
        "calibration_version",
        "threshold_policy_version",
        "replay_mode",
        "ood_state",
        "suppression_reason",
        "probability_contract_json",
    }.issubset(pred_cols)
    assert {"source_publish_time_utc", "source_revision"}.issubset(raw_cols)


def test_prediction_upsert_is_idempotent_and_stable(tmp_path):
    db = DbWriter(str(tmp_path / "idempotent.duckdb"), str(tmp_path / "idempotent.lock"))
    snapshot_id, _ = _bootstrap_snapshot(db)

    payload = _prediction_payload(snapshot_id)
    with db.writer() as con:
        first_id = db.insert_prediction(con, payload)
        second_id = db.insert_prediction(
            con,
            _prediction_payload(snapshot_id, confidence=0.72, prob_up=0.7, model_hash="hash_v2"),
        )
        rows = con.execute(
            """
            SELECT prediction_id, prediction_business_key, confidence, prob_up, model_hash,
                   target_name, target_version, label_version, feature_version,
                   calibration_version, threshold_policy_version, replay_mode,
                   ood_state, suppression_reason, probability_contract_json
            FROM predictions
            """
        ).fetchall()

    assert first_id == second_id
    assert len(rows) == 1
    (
        prediction_id,
        business_key,
        confidence,
        prob_up,
        model_hash,
        target_name,
        target_version,
        label_version,
        feature_version,
        calibration_version,
        threshold_policy_version,
        replay_mode,
        ood_state,
        suppression_reason,
        probability_contract_json,
    ) = rows[0]

    assert str(prediction_id) == first_id
    assert business_key
    assert confidence == 0.72
    assert prob_up == 0.7
    assert model_hash == "hash_v2"
    assert target_name == "intraday_direction_3class"
    assert target_version == "target_v1"
    assert label_version == "label_v2"
    assert feature_version.startswith("derived_feature_contract_")
    assert calibration_version == "cal_v7"
    assert threshold_policy_version == "threshold_v3"
    assert replay_mode == "UNKNOWN"
    assert ood_state == "UNKNOWN"
    assert suppression_reason is None

    parsed_contract = json.loads(probability_contract_json) if isinstance(probability_contract_json, str) else probability_contract_json
    assert parsed_contract["calibration_artifact_ref"]["artifact_version"] == "cal_v7"


def test_migration_backfills_prediction_and_raw_event_lineage(tmp_path):
    db_path = tmp_path / "migrate.duckdb"
    con = duckdb.connect(str(db_path))
    con.execute(
        """
        CREATE TABLE predictions (
            prediction_id VARCHAR PRIMARY KEY,
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
        CREATE TABLE raw_http_events (
            event_id VARCHAR PRIMARY KEY,
            run_id VARCHAR,
            requested_at_utc TIMESTAMP,
            received_at_utc TIMESTAMP,
            ticker TEXT,
            endpoint_id INTEGER,
            http_status INTEGER,
            latency_ms INTEGER,
            payload_hash TEXT,
            payload_json JSON,
            is_retry BOOLEAN,
            error_type TEXT,
            error_msg TEXT,
            circuit_state_json JSON
        )
        """
    )
    legacy_meta = {
        "prediction_contract": {
            "target_name": "intraday_direction_3class",
            "target_version": "legacy_target_v1",
            "label_version": "legacy_label_v2",
            "threshold_policy_version": "legacy_threshold_v3",
        },
        "probability_contract": {
            "target_spec": {
                "target_name": "intraday_direction_3class",
                "target_version": "legacy_target_v1",
            },
            "calibration_artifact_ref": {
                "artifact_version": "legacy_cal_v4",
            },
            "ood_state": "DEGRADED",
            "suppression_reason": "MISSING_CALIBRATION_ARTIFACT",
        },
        "freshness_registry_diagnostics": {
            "feature_policies": {
                "spot": {
                    "policy": "REQUIRED_LIVE",
                    "lag_class": "LIVE",
                    "join_skew_tolerance_seconds": 60,
                    "max_tolerated_age_seconds": 60,
                    "policy_source": "registry",
                }
            }
        },
        "replay_mode": "UNKNOWN",
    }
    con.execute(
        """
        INSERT INTO predictions VALUES (
            'pred-legacy', 'snap-legacy', 15, 'FIXED', NULL,
            100.0, 0.2, 0.5, 0.6, 0.2, 0.2,
            'model', 'legacy_model_v1', 'hash1', false,
            false, NULL, NULL, NULL, NULL, NULL, NULL,
            ?, 'LONG', 'PASS', 'VALID', 'MEDIUM', '[]', '[]', true,
            '{}', 'ALIGNED', NULL, NULL, 0, 'legacy_window'
        )
        """,
        [json.dumps(legacy_meta)],
    )
    con.execute(
        """
        INSERT INTO raw_http_events VALUES (
            'event-legacy', 'run-legacy', ?, ?, 'AAPL', 1,
            200, 10, 'ph', ?, false, NULL, NULL, '{}'
        )
        """,
        [
            datetime(2026, 1, 2, 15, 30, tzinfo=UTC),
            datetime(2026, 1, 2, 15, 30, tzinfo=UTC),
            json.dumps({"published_at": "2026-01-02T15:25:00Z", "revision": "rev-9"}),
        ],
    )
    con.close()

    db = DbWriter(str(db_path), str(tmp_path / "migrate.lock"))
    with db.writer() as con2:
        db.ensure_schema(con2)
        row = con2.execute(
            """
            SELECT prediction_business_key, target_name, target_version, label_version,
                   feature_version, calibration_version, threshold_policy_version,
                   replay_mode, ood_state, suppression_reason, probability_contract_json
            FROM predictions WHERE prediction_id = 'pred-legacy'
            """
        ).fetchone()
        raw = con2.execute(
            "SELECT source_publish_time_utc, source_revision FROM raw_http_events WHERE event_id = 'event-legacy'"
        ).fetchone()

    assert row[0]
    assert row[1] == "intraday_direction_3class"
    assert row[2] == "legacy_target_v1"
    assert row[3] == "legacy_label_v2"
    assert row[4].startswith("derived_feature_contract_")
    assert row[5] == "legacy_cal_v4"
    assert row[6] == "legacy_threshold_v3"
    assert row[7] == "UNKNOWN"
    assert row[8] == "DEGRADED"
    assert row[9] == "MISSING_CALIBRATION_ARTIFACT"
    parsed_contract = json.loads(row[10]) if isinstance(row[10], str) else row[10]
    assert parsed_contract["calibration_artifact_ref"]["artifact_version"] == "legacy_cal_v4"

    assert raw[0] is not None
    assert raw[1] == "rev-9"


def test_existing_validator_style_read_query_remains_compatible(tmp_path):
    db = DbWriter(str(tmp_path / "compat.duckdb"), str(tmp_path / "compat.lock"))
    snapshot_id, now = _bootstrap_snapshot(db)

    with db.writer() as con:
        db.insert_prediction(con, _prediction_payload(snapshot_id))
        rows = con.execute(
            """
            SELECT p.prediction_id, p.snapshot_id, p.horizon_kind, p.horizon_minutes, p.horizon_seconds,
                   p.start_price, p.prob_up, p.prob_down, p.prob_flat, s.ticker, s.asof_ts_utc,
                   p.decision_state, p.decision_window_id, p.validation_eligible, p.meta_json,
                   s.market_close_utc, s.post_end_utc, s.is_early_close, s.session_label
            FROM predictions p
            JOIN snapshots s ON s.snapshot_id = p.snapshot_id
            WHERE s.asof_ts_utc = ?
            """,
            [now],
        ).fetchall()

    assert len(rows) == 1
    assert rows[0][2] == "FIXED"
    assert rows[0][9] == "AAPL"
