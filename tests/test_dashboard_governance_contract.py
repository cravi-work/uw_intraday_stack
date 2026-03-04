from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

import duckdb
import pandas as pd
import pytest

from src.dashboard_app import build_decision_trace_frame, build_prediction_contract_frame
from src.storage import DbWriter

UTC = timezone.utc


def _insert_prediction(
    con: duckdb.DuckDBPyConnection,
    *,
    snapshot_id,
    horizon_minutes: int,
    probability_contract: dict,
    ood_state: str,
    ood_reason: str | None,
    calibration_version: str | None,
    calibration_scope: dict | None,
    calibration_artifact_hash: str | None,
    suppression_reason: str | None,
    data_quality_state: str,
    confidence_state: str,
    meta_json: dict | None = None,
) -> None:
    con.execute(
        """
        INSERT INTO predictions (
            prediction_id, prediction_business_key, snapshot_id,
            horizon_minutes, horizon_kind, horizon_seconds,
            bias, confidence, prob_up, prob_down, prob_flat,
            target_name, target_version, label_version,
            model_name, model_version, calibration_version, threshold_policy_version,
            replay_mode, ood_state, ood_reason, calibration_scope, calibration_artifact_hash,
            decision_path_contract_version, suppression_reason, probability_contract_json, meta_json,
            outcome_realized, is_mock, realized_at_utc, brier_score,
            decision_state, risk_gate_status, data_quality_state, confidence_state
        ) VALUES (?, ?, ?, ?, 'FIXED', ?, 0.10, 0.66, 0.58, 0.21, 0.21,
                  'intraday_direction_3class', 'target_v2', 'label_v2',
                  'phase0_additive', '2.0', ?, 'thresh_v3',
                  'LIVE_LIKE_OBSERVED', ?, ?, ?, ?,
                  'decision_path/v2', ?, ?, ?,
                  FALSE, FALSE, NULL, NULL,
                  'LONG', 'PASS', ?, ?)
        """,
        [
            uuid.uuid4(),
            f"{snapshot_id}:{horizon_minutes}",
            snapshot_id,
            horizon_minutes,
            horizon_minutes * 60,
            calibration_version,
            ood_state,
            ood_reason,
            json.dumps(calibration_scope) if calibration_scope is not None else None,
            calibration_artifact_hash,
            suppression_reason,
            json.dumps(probability_contract),
            json.dumps(meta_json or {}),
            data_quality_state,
            confidence_state,
        ],
    )


def _seed_governance_dashboard_db(tmp_path):
    db_path = tmp_path / "dashboard_governance.duckdb"
    writer = DbWriter(str(db_path))
    con = duckdb.connect(str(db_path))
    writer.ensure_schema(con)

    snapshot_id = uuid.uuid4()
    asof_ts = datetime(2026, 3, 4, 15, 30, tzinfo=UTC)

    con.execute("INSERT INTO dim_tickers (ticker) VALUES ('MSFT')")
    con.execute(
        """
        INSERT INTO snapshots (
            snapshot_id, asof_ts_utc, ticker, session_label, data_quality_score,
            market_close_utc, post_end_utc, seconds_to_close, is_early_close
        ) VALUES (?, ?, 'MSFT', 'REG', 0.94, ?, ?, 1500, FALSE)
        """,
        [
            snapshot_id,
            asof_ts,
            datetime(2026, 3, 4, 21, 0, tzinfo=UTC),
            datetime(2026, 3, 5, 1, 0, tzinfo=UTC),
        ],
    )

    shared_scope = {
        "horizon_kind": "FIXED",
        "session": "RTH",
        "regime": "NORMAL",
        "replay_mode": "LIVE_LIKE_OBSERVED",
        "scope_contract_version": "calibration_scope/v1",
    }

    _insert_prediction(
        con,
        snapshot_id=snapshot_id,
        horizon_minutes=5,
        probability_contract={
            "calibrated_probability_vector": {"UP": 0.58, "DOWN": 0.21, "FLAT": 0.21},
            "suppression_reason": None,
            "ood_state": "DEGRADED",
            "ood_reason": "FEATURE_COVERAGE_BELOW_TARGET",
            "is_coherent": True,
            "calibration_artifact_ref": {
                "artifact_version": "cal_v8",
                "artifact_hash": "hash_v8",
                "calibration_scope": {**shared_scope, "horizon_minutes": 5},
            },
        },
        ood_state="DEGRADED",
        ood_reason="FEATURE_COVERAGE_BELOW_TARGET",
        calibration_version="cal_v8",
        calibration_scope={**shared_scope, "horizon_minutes": 5},
        calibration_artifact_hash="hash_v8",
        suppression_reason=None,
        data_quality_state="PARTIAL",
        confidence_state="DEGRADED",
        meta_json={
            "prediction_contract": {"decision_path_contract_version": "decision_path/v2"},
            "horizon_contract": {
                "decision_path_contract_version": "decision_path/v2",
                "report_only_excluded_features": ["darkpool_pressure"],
                "context_only_excluded_features": ["iv_rank"],
            },
        },
    )

    _insert_prediction(
        con,
        snapshot_id=snapshot_id,
        horizon_minutes=10,
        probability_contract={
            "calibrated_probability_vector": {"UP": 0.57, "DOWN": 0.22, "FLAT": 0.21},
            "suppression_reason": None,
            "ood_state": "UNKNOWN",
            "ood_reason": "ASSESSMENT_SKIPPED",
            "is_coherent": True,
        },
        ood_state="UNKNOWN",
        ood_reason="ASSESSMENT_SKIPPED",
        calibration_version="cal_v8",
        calibration_scope=None,
        calibration_artifact_hash=None,
        suppression_reason=None,
        data_quality_state="OK",
        confidence_state="HIGH",
        meta_json={
            "prediction_contract": {"decision_path_contract_version": "decision_path/v2"},
            "horizon_contract": {"decision_path_contract_version": "decision_path/v2"},
        },
    )

    trace_json = {
        "ood_reason": "FEATURE_COVERAGE_BELOW_TARGET",
        "probability_contract": {
            "calibration_artifact_ref": {
                "artifact_hash": "hash_v8",
                "calibration_scope": {**shared_scope, "horizon_minutes": 5},
            }
        },
        "horizon_contract": {
            "decision_path_contract_version": "decision_path/v2",
            "report_only_excluded_features": ["darkpool_pressure"],
            "context_only_excluded_features": ["iv_rank"],
        },
    }
    con.execute(
        """
        INSERT INTO decision_traces (
            trace_id, created_at_utc, prediction_id, prediction_business_key, snapshot_id,
            event_type, decision_state, risk_gate_status, data_quality_state,
            confidence_state, suppression_reason, ood_state, ood_reason, replay_mode,
            model_version, target_version, calibration_version,
            calibration_scope, calibration_artifact_hash, decision_path_contract_version,
            trace_json
        ) VALUES (?, ?, NULL, ?, ?,
                  'SIGNAL_DEGRADED', 'LONG', 'DEGRADED', 'PARTIAL',
                  'DEGRADED', NULL, 'DEGRADED', 'FEATURE_COVERAGE_BELOW_TARGET', 'LIVE_LIKE_OBSERVED',
                  '2.0', 'target_v2', 'cal_v8',
                  ?, 'hash_v8', 'decision_path/v2', ?)
        """,
        [
            uuid.uuid4(),
            datetime(2026, 3, 4, 15, 31, tzinfo=UTC),
            f"{snapshot_id}:5",
            snapshot_id,
            json.dumps({**shared_scope, "horizon_minutes": 5}),
            json.dumps(trace_json),
        ],
    )

    return con, str(snapshot_id)


def test_prediction_frame_distinguishes_degraded_and_unknown_governance_states(tmp_path):
    con, snapshot_id = _seed_governance_dashboard_db(tmp_path)
    try:
        df = build_prediction_contract_frame(con, snapshot_id)
    finally:
        con.close()

    degraded = df.loc[df["horizon_minutes"] == 5].iloc[0]
    assert degraded["governance_state"] == "DEGRADED"
    assert degraded["governance_reason"] == "FEATURE_COVERAGE_BELOW_TARGET, QUALITY_PARTIAL, CONFIDENCE_DEGRADED"
    assert degraded["calibrated_prob_up"] == pytest.approx(0.58)
    assert "session=RTH" in str(degraded["calibration_scope_label"])
    assert "report_only=darkpool_pressure" in str(degraded["decision_path_exclusions_text"])
    assert "context_only=iv_rank" in str(degraded["decision_path_exclusions_text"])

    unknown = df.loc[df["horizon_minutes"] == 10].iloc[0]
    assert unknown["governance_state"] == "UNKNOWN_GOVERNANCE"
    assert "OOD_UNKNOWN" in str(unknown["governance_reason"])
    assert "CALIBRATION_SCOPE_MISSING" in str(unknown["governance_reason"])
    assert "CALIBRATION_ARTIFACT_HASH_MISSING" in str(unknown["governance_reason"])
    assert pd.isna(unknown["calibrated_prob_up"])
    assert pd.isna(unknown["calibrated_prob_down"])
    assert pd.isna(unknown["calibrated_prob_flat"])


def test_decision_trace_frame_surfaces_governance_details_truthfully(tmp_path):
    con, snapshot_id = _seed_governance_dashboard_db(tmp_path)
    try:
        df = build_decision_trace_frame(con, snapshot_id)
    finally:
        con.close()

    row = df.iloc[0]
    assert row["governance_state"] == "DEGRADED"
    assert row["ood_reason"] == "FEATURE_COVERAGE_BELOW_TARGET"
    assert "session=RTH" in str(row["calibration_scope_label"])
    assert row["decision_path_contract_version"] == "decision_path/v2"
    assert "report_only=darkpool_pressure" in str(row["decision_path_exclusions_text"])
    assert "context_only=iv_rank" in str(row["decision_path_exclusions_text"])
