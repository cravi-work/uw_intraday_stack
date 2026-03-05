from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

import duckdb
import pandas as pd
import pytest

from src.dashboard_app import (
    DASHBOARD_DISCLAIMER,
    PAGE_TITLE,
    build_calibrated_scorecard_frame,
    build_decision_trace_frame,
    build_prediction_contract_frame,
    build_realized_prediction_contract_frame,
)
from src.storage import DbWriter

UTC = timezone.utc


def _insert_feature_lineage(con: duckdb.DuckDBPyConnection, snapshot_id) -> None:
    con.execute(
        """
        INSERT INTO features (snapshot_id, feature_key, feature_value, meta_json)
        VALUES
          (?, 'spot', 197.25, ?),
          (?, 'oi_pressure', 0.31, ?)
        """,
        [
            snapshot_id,
            json.dumps(
                {
                    "metric_lineage": {
                        "metric_name": "spot",
                        "timestamp_quality": "VALID",
                        "time_provenance_degraded": False,
                        "bounded_output": False,
                    }
                }
            ),
            snapshot_id,
            json.dumps(
                {
                    "metric_lineage": {
                        "metric_name": "oi_pressure",
                        "timestamp_quality": "DEGRADED",
                        "time_provenance_degraded": True,
                        "bounded_output": True,
                        "output_domain_contract_version": "output_domain/v1",
                    }
                }
            ),
        ],
    )


def _seed_dashboard_db(tmp_path):
    db_path = tmp_path / "dashboard.duckdb"
    writer = DbWriter(str(db_path))
    con = duckdb.connect(str(db_path))
    writer.ensure_schema(con)

    snapshot_id = uuid.uuid4()
    prediction_id_good = uuid.uuid4()
    prediction_id_suppressed = uuid.uuid4()
    asof_ts = datetime(2026, 3, 3, 15, 30, tzinfo=UTC)

    con.execute("INSERT INTO dim_tickers (ticker) VALUES ('AAPL')")
    con.execute(
        """
        INSERT INTO snapshots (
            snapshot_id, asof_ts_utc, ticker, session_label, data_quality_score,
            market_close_utc, post_end_utc, seconds_to_close, is_early_close
        ) VALUES (?, ?, 'AAPL', 'REG', 0.97, ?, ?, 1800, FALSE)
        """,
        [
            snapshot_id,
            asof_ts,
            datetime(2026, 3, 3, 21, 0, tzinfo=UTC),
            datetime(2026, 3, 4, 1, 0, tzinfo=UTC),
        ],
    )
    _insert_feature_lineage(con, snapshot_id)

    good_scope = {
        "horizon_kind": "FIXED",
        "horizon_minutes": 15,
        "session": "RTH",
        "regime": "NORMAL",
        "replay_mode": "LIVE_LIKE_OBSERVED",
        "scope_contract_version": "calibration_scope/v1",
    }
    good_contract = {
        "calibrated_probability_vector": {"UP": 0.62, "DOWN": 0.18, "FLAT": 0.20},
        "suppression_reason": None,
        "ood_state": "IN_DISTRIBUTION",
        "ood_reason": None,
        "is_coherent": True,
        "calibration_artifact_ref": {
            "artifact_version": "cal_v7",
            "artifact_hash": "hash_cal_v7",
            "calibration_scope": good_scope,
            "artifact_provenance": {
                "evidence_ref": "evidence://calibration/report-v7",
            },
        },
        "ood_contract_version": "ood/v2",
        "output_domain_contract_version": "output_domain/v1",
    }
    suppressed_contract = {
        "calibrated_probability_vector": None,
        "suppression_reason": "MISSING_CALIBRATION_ARTIFACT",
        "ood_state": "UNKNOWN",
        "ood_reason": "CALIBRATION_SELECTION_FAILED",
        "is_coherent": False,
        "ood_contract_version": "ood/v2",
        "output_domain_contract_version": "output_domain/v1",
    }

    good_meta = {
        "prediction_contract": {"decision_path_contract_version": "decision_path/v2"},
        "horizon_contract": {
            "decision_path_contract_version": "decision_path/v2",
            "report_only_excluded_features": ["darkpool_pressure"],
        },
        "replay_governance": {
            "requested_replay_mode": "LIVE_LIKE_OBSERVED",
            "prediction_replay_mode": "LIVE_LIKE_OBSERVED",
            "calibration_request_replay_mode": "LIVE_LIKE_OBSERVED",
            "calibration_artifact_scope_replay_mode": "LIVE_LIKE_OBSERVED",
            "calibration_selection_reason": "REPLAY_MODE_COMPATIBLE",
        },
        "ood_assessment": {"contract_version": "ood/v2"},
        "output_domain_contract_version": "output_domain/v1",
    }
    suppressed_meta = {
        "prediction_contract": {"decision_path_contract_version": "decision_path/v2"},
        "horizon_contract": {
            "decision_path_contract_version": "decision_path/v2",
            "context_only_excluded_features": ["iv_rank"],
        },
        "ood_assessment": {"contract_version": "ood/v2"},
        "output_domain_contract_version": "output_domain/v1",
    }

    con.execute(
        """
        INSERT INTO predictions (
            prediction_id, prediction_business_key, snapshot_id,
            horizon_minutes, horizon_kind, horizon_seconds,
            bias, confidence, prob_up, prob_down, prob_flat,
            target_name, target_version, label_version,
            model_name, model_version, calibration_version, threshold_policy_version,
            replay_mode, ood_state, ood_reason, calibration_scope, calibration_artifact_hash,
            calibration_evidence_ref, output_domain_contract_version, replay_governance_reason,
            ood_contract_version, decision_path_contract_version, suppression_reason,
            probability_contract_json, meta_json,
            outcome_realized, is_mock, realized_at_utc, brier_score,
            decision_state, risk_gate_status, data_quality_state, confidence_state
        ) VALUES (?, ?, ?, 15, 'FIXED', 900, 0.14, 0.71, 0.62, 0.18, 0.20,
                  'intraday_direction_3class', 'target_v2', 'label_v2',
                  'phase0_additive', '2.0', 'cal_v7', 'thresh_v3',
                  'LIVE_LIKE_OBSERVED', 'IN_DISTRIBUTION', NULL, ?, ?,
                  'evidence://calibration/report-v7', 'output_domain/v1', 'REPLAY_MODE_COMPATIBLE',
                  'ood/v2', 'decision_path/v2', NULL,
                  ?, ?,
                  TRUE, FALSE, ?, 0.12,
                  'LONG', 'PASS', 'OK', 'HIGH')
        """,
        [
            prediction_id_good,
            f"{snapshot_id}:15",
            snapshot_id,
            json.dumps(good_scope),
            "hash_cal_v7",
            json.dumps(good_contract),
            json.dumps(good_meta),
            datetime(2026, 3, 3, 15, 45, tzinfo=UTC),
        ],
    )

    con.execute(
        """
        INSERT INTO predictions (
            prediction_id, prediction_business_key, snapshot_id,
            horizon_minutes, horizon_kind, horizon_seconds,
            bias, confidence, prob_up, prob_down, prob_flat,
            target_name, target_version, label_version,
            model_name, model_version, calibration_version, threshold_policy_version,
            replay_mode, ood_state, ood_reason, calibration_scope, calibration_artifact_hash,
            calibration_evidence_ref, output_domain_contract_version, replay_governance_reason,
            ood_contract_version, decision_path_contract_version, suppression_reason,
            probability_contract_json, meta_json,
            outcome_realized, is_mock, realized_at_utc, brier_score,
            decision_state, risk_gate_status, data_quality_state, confidence_state
        ) VALUES (?, ?, ?, 30, 'FIXED', 1800, 0.09, 0.45, 0.55, 0.20, 0.25,
                  'intraday_direction_3class', 'target_v2', 'label_v2',
                  'phase0_additive', '2.0', NULL, 'thresh_v3',
                  'LIVE_LIKE_OBSERVED', 'UNKNOWN', 'CALIBRATION_SELECTION_FAILED', NULL, NULL,
                  NULL, 'output_domain/v1', NULL,
                  'ood/v2', 'decision_path/v2', 'MISSING_CALIBRATION_ARTIFACT',
                  ?, ?,
                  TRUE, FALSE, ?, 0.33,
                  'NO_SIGNAL', 'BLOCKED', 'PARTIAL', 'LOW')
        """,
        [
            prediction_id_suppressed,
            f"{snapshot_id}:30",
            snapshot_id,
            json.dumps(suppressed_contract),
            json.dumps(suppressed_meta),
            datetime(2026, 3, 3, 16, 0, tzinfo=UTC),
        ],
    )

    trace_json = {
        "ood_reason": None,
        "calibration_evidence_ref": "evidence://calibration/report-v7",
        "output_domain_contract_version": "output_domain/v1",
        "ood_contract_version": "ood/v2",
        "replay_governance_reason": "REPLAY_MODE_COMPATIBLE",
        "probability_contract": {
            "calibration_artifact_ref": {
                "artifact_hash": "hash_cal_v7",
                "calibration_scope": good_scope,
                "artifact_provenance": {"evidence_ref": "evidence://calibration/report-v7"},
            }
        },
        "horizon_contract": {
            "decision_path_contract_version": "decision_path/v2",
            "report_only_excluded_features": ["darkpool_pressure"],
        },
    }
    con.execute(
        """
        INSERT INTO decision_traces (
            trace_id, created_at_utc, prediction_id, prediction_business_key, snapshot_id,
            event_type, decision_state, risk_gate_status, data_quality_state,
            confidence_state, suppression_reason, ood_state, ood_reason, replay_mode,
            model_version, target_version, calibration_version,
            calibration_scope, calibration_artifact_hash, calibration_evidence_ref,
            output_domain_contract_version, replay_governance_reason, ood_contract_version,
            decision_path_contract_version, trace_json
        ) VALUES (?, ?, ?, ?, ?,
                  'SIGNAL_EMITTED', 'LONG', 'PASS', 'OK',
                  'HIGH', NULL, 'IN_DISTRIBUTION', NULL, 'LIVE_LIKE_OBSERVED',
                  '2.0', 'target_v2', 'cal_v7',
                  ?, 'hash_cal_v7', 'evidence://calibration/report-v7',
                  'output_domain/v1', 'REPLAY_MODE_COMPATIBLE', 'ood/v2',
                  'decision_path/v2', ?)
        """,
        [
            uuid.uuid4(),
            datetime(2026, 3, 3, 15, 31, tzinfo=UTC),
            prediction_id_good,
            f"{snapshot_id}:15",
            snapshot_id,
            json.dumps(good_scope),
            json.dumps(trace_json),
        ],
    )

    return con, str(snapshot_id)


def test_prediction_contract_frame_surfaces_governance_audit_fields_and_hides_suppressed_probabilities(tmp_path):
    con, snapshot_id = _seed_dashboard_db(tmp_path)
    try:
        df = build_prediction_contract_frame(con, snapshot_id)
    finally:
        con.close()

    assert {
        "target_version",
        "calibration_version",
        "calibration_scope_label",
        "calibration_artifact_hash",
        "calibration_evidence_ref",
        "replay_mode",
        "replay_governance_reason",
        "ood_state",
        "ood_reason",
        "ood_contract_version",
        "output_domain_contract_version",
        "suppression_reason",
        "quality_state",
        "probability_state",
        "governance_reason",
        "decision_path_exclusions_text",
        "time_provenance_degraded_count",
        "time_provenance_degraded_features_text",
        "calibrated_prob_up",
        "calibrated_prob_down",
        "calibrated_prob_flat",
    }.issubset(df.columns)

    good = df.loc[df["horizon_minutes"] == 15].iloc[0]
    assert good["target_version"] == "target_v2"
    assert good["calibration_version"] == "cal_v7"
    assert good["replay_mode"] == "LIVE_LIKE_OBSERVED"
    assert good["replay_governance_reason"] == "REPLAY_MODE_COMPATIBLE"
    assert good["calibration_artifact_hash"] == "hash_cal_v7"
    assert good["calibration_evidence_ref"] == "evidence://calibration/report-v7"
    assert good["ood_contract_version"] == "ood/v2"
    assert good["output_domain_contract_version"] == "output_domain/v1"
    assert good["quality_state"] == "OK"
    assert good["probability_state"] == "CALIBRATED"
    assert good["calibrated_prob_up"] == pytest.approx(0.62)
    assert good["calibrated_prob_down"] == pytest.approx(0.18)
    assert good["calibrated_prob_flat"] == pytest.approx(0.20)
    assert "session=RTH" in str(good["calibration_scope_label"])
    assert "report_only=darkpool_pressure" in str(good["decision_path_exclusions_text"])
    assert int(good["time_provenance_degraded_count"]) == 1
    assert "oi_pressure" in str(good["time_provenance_degraded_features_text"])

    suppressed = df.loc[df["horizon_minutes"] == 30].iloc[0]
    assert suppressed["suppression_reason"] == "MISSING_CALIBRATION_ARTIFACT"
    assert suppressed["quality_state"] == "PARTIAL"
    assert suppressed["probability_state"] == "SUPPRESSED"
    assert suppressed["ood_reason"] == "CALIBRATION_SELECTION_FAILED"
    assert pd.isna(suppressed["calibrated_prob_up"])
    assert pd.isna(suppressed["calibrated_prob_down"])
    assert pd.isna(suppressed["calibrated_prob_flat"])


def test_decision_trace_frame_includes_reporting_governance_audit_fields(tmp_path):
    con, snapshot_id = _seed_dashboard_db(tmp_path)
    try:
        df = build_decision_trace_frame(con, snapshot_id)
    finally:
        con.close()

    assert {
        "target_version",
        "calibration_version",
        "calibration_scope_label",
        "calibration_artifact_hash",
        "calibration_evidence_ref",
        "replay_mode",
        "replay_governance_reason",
        "ood_state",
        "ood_reason",
        "ood_contract_version",
        "output_domain_contract_version",
        "suppression_reason",
        "quality_state",
        "governance_state",
        "decision_path_exclusions_text",
        "time_provenance_degraded_count",
        "time_provenance_degraded_features_text",
    }.issubset(df.columns)

    row = df.iloc[0]
    assert row["replay_mode"] == "LIVE_LIKE_OBSERVED"
    assert row["replay_governance_reason"] == "REPLAY_MODE_COMPATIBLE"
    assert row["quality_state"] == "OK"
    assert row["governance_state"] == "CALIBRATED"
    assert row["calibration_artifact_hash"] == "hash_cal_v7"
    assert row["calibration_evidence_ref"] == "evidence://calibration/report-v7"
    assert row["ood_contract_version"] == "ood/v2"
    assert row["output_domain_contract_version"] == "output_domain/v1"
    assert "report_only=darkpool_pressure" in str(row["decision_path_exclusions_text"])
    assert int(row["time_provenance_degraded_count"]) == 1
    assert "oi_pressure" in str(row["time_provenance_degraded_features_text"])


def test_scorecard_uses_only_coherent_fully_governed_calibrated_probability_rows(tmp_path):
    con, snapshot_id = _seed_dashboard_db(tmp_path)
    try:
        realized = build_realized_prediction_contract_frame(con)
        agg = build_calibrated_scorecard_frame(con)
    finally:
        con.close()

    assert len(realized) == 2
    assert set(realized["probability_state"]) == {"CALIBRATED", "SUPPRESSED"}

    assert len(agg) == 1
    row = agg.iloc[0]
    assert row["session_label"] == "REG"
    assert int(row["preds"]) == 1
    assert float(row["mean_brier"]) == pytest.approx(0.12)


def test_dashboard_copy_does_not_imply_live_trade_readiness():
    assert PAGE_TITLE == "Decision Support Dashboard"
    assert "institutional" not in PAGE_TITLE.lower()
    assert "does not imply live-trade readiness" in DASHBOARD_DISCLAIMER.lower()
    assert "governance state is shown explicitly" in DASHBOARD_DISCLAIMER.lower()
