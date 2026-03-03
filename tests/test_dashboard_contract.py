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

    good_contract = {
        "calibrated_probability_vector": {"UP": 0.62, "DOWN": 0.18, "FLAT": 0.20},
        "suppression_reason": None,
        "ood_state": "UNKNOWN",
        "is_coherent": True,
    }
    suppressed_contract = {
        "calibrated_probability_vector": None,
        "suppression_reason": "MISSING_CALIBRATION_ARTIFACT",
        "ood_state": "UNKNOWN",
        "is_coherent": False,
    }

    con.execute(
        """
        INSERT INTO predictions (
            prediction_id, prediction_business_key, snapshot_id,
            horizon_minutes, horizon_kind, horizon_seconds,
            bias, confidence, prob_up, prob_down, prob_flat,
            target_name, target_version, label_version,
            model_name, model_version, calibration_version, threshold_policy_version,
            replay_mode, ood_state, suppression_reason, probability_contract_json,
            outcome_realized, is_mock, realized_at_utc, brier_score,
            decision_state, risk_gate_status, data_quality_state, confidence_state
        ) VALUES (?, ?, ?, 15, 'FIXED', 900, 0.14, 0.71, 0.62, 0.18, 0.20,
                  'intraday_direction_3class', 'target_v2', 'label_v2',
                  'phase0_additive', '2.0', 'cal_v7', 'thresh_v3',
                  'LIVE_LIKE_OBSERVED', 'UNKNOWN', NULL, ?,
                  TRUE, FALSE, ?, 0.12,
                  'LONG', 'PASS', 'OK', 'HIGH')
        """,
        [
            prediction_id_good,
            f"{snapshot_id}:15",
            snapshot_id,
            json.dumps(good_contract),
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
            replay_mode, ood_state, suppression_reason, probability_contract_json,
            outcome_realized, is_mock, realized_at_utc, brier_score,
            decision_state, risk_gate_status, data_quality_state, confidence_state
        ) VALUES (?, ?, ?, 30, 'FIXED', 1800, 0.09, 0.45, 0.55, 0.20, 0.25,
                  'intraday_direction_3class', 'target_v2', 'label_v2',
                  'phase0_additive', '2.0', NULL, 'thresh_v3',
                  'LIVE_LIKE_OBSERVED', 'UNKNOWN', 'MISSING_CALIBRATION_ARTIFACT', ?,
                  TRUE, FALSE, ?, 0.33,
                  'NO_SIGNAL', 'BLOCKED', 'PARTIAL', 'LOW')
        """,
        [
            prediction_id_suppressed,
            f"{snapshot_id}:30",
            snapshot_id,
            json.dumps(suppressed_contract),
            datetime(2026, 3, 3, 16, 0, tzinfo=UTC),
        ],
    )

    con.execute(
        """
        INSERT INTO decision_traces (
            trace_id, created_at_utc, prediction_id, prediction_business_key, snapshot_id,
            event_type, decision_state, risk_gate_status, data_quality_state,
            confidence_state, suppression_reason, ood_state, replay_mode,
            model_version, target_version, calibration_version, trace_json
        ) VALUES (?, ?, ?, ?, ?,
                  'SUPPRESSED_SIGNAL', 'NO_SIGNAL', 'BLOCKED', 'PARTIAL',
                  'LOW', 'MISSING_CALIBRATION_ARTIFACT', 'UNKNOWN', 'LIVE_LIKE_OBSERVED',
                  '2.0', 'target_v2', NULL, '{}')
        """,
        [
            uuid.uuid4(),
            datetime(2026, 3, 3, 15, 31, tzinfo=UTC),
            prediction_id_suppressed,
            f"{snapshot_id}:30",
            snapshot_id,
        ],
    )

    return con, str(snapshot_id)


def test_prediction_contract_frame_surfaces_contract_fields_and_hides_suppressed_probabilities(tmp_path):
    con, snapshot_id = _seed_dashboard_db(tmp_path)
    try:
        df = build_prediction_contract_frame(con, snapshot_id)
    finally:
        con.close()

    assert {
        "target_version",
        "calibration_version",
        "replay_mode",
        "ood_state",
        "suppression_reason",
        "quality_state",
        "probability_state",
        "calibrated_prob_up",
        "calibrated_prob_down",
        "calibrated_prob_flat",
    }.issubset(df.columns)

    good = df.loc[df["horizon_minutes"] == 15].iloc[0]
    assert good["target_version"] == "target_v2"
    assert good["calibration_version"] == "cal_v7"
    assert good["replay_mode"] == "LIVE_LIKE_OBSERVED"
    assert good["quality_state"] == "OK"
    assert good["probability_state"] == "CALIBRATED"
    assert good["calibrated_prob_up"] == pytest.approx(0.62)
    assert good["calibrated_prob_down"] == pytest.approx(0.18)
    assert good["calibrated_prob_flat"] == pytest.approx(0.20)

    suppressed = df.loc[df["horizon_minutes"] == 30].iloc[0]
    assert suppressed["suppression_reason"] == "MISSING_CALIBRATION_ARTIFACT"
    assert suppressed["quality_state"] == "PARTIAL"
    assert suppressed["probability_state"] == "SUPPRESSED"
    assert pd.isna(suppressed["calibrated_prob_up"])
    assert pd.isna(suppressed["calibrated_prob_down"])
    assert pd.isna(suppressed["calibrated_prob_flat"])


def test_decision_trace_frame_includes_reporting_contract_context(tmp_path):
    con, snapshot_id = _seed_dashboard_db(tmp_path)
    try:
        df = build_decision_trace_frame(con, snapshot_id)
    finally:
        con.close()

    assert {
        "target_version",
        "calibration_version",
        "replay_mode",
        "ood_state",
        "suppression_reason",
        "quality_state",
    }.issubset(df.columns)

    row = df.iloc[0]
    assert row["replay_mode"] == "LIVE_LIKE_OBSERVED"
    assert row["suppression_reason"] == "MISSING_CALIBRATION_ARTIFACT"
    assert row["quality_state"] == "PARTIAL"


def test_scorecard_uses_only_coherent_calibrated_probability_rows(tmp_path):
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
