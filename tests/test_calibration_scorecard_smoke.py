from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import duckdb
import pandas as pd

from scripts.calibration_scorecard import generate_calibration_scorecard
from src.storage import DbWriter

UTC = timezone.utc


def _seed_snapshot(db: DbWriter, con: duckdb.DuckDBPyConnection, *, ticker: str, asof_ts: datetime, session_label: str) -> str:
    cfg_ver = db.insert_config(con, f"ticker: {ticker}\n")
    run_id = db.begin_run(
        con,
        asof_ts_utc=asof_ts,
        session_label=session_label,
        is_trading_day=True,
        is_early_close=False,
        config_version=cfg_ver,
        api_catalog_hash="catalog_hash_p2401",
    )
    db.upsert_tickers(con, [ticker])
    return db.insert_snapshot(
        con,
        run_id=run_id,
        asof_ts_utc=asof_ts,
        ticker=ticker,
        session_label=session_label,
        is_trading_day=True,
        is_early_close=False,
        data_quality_score=0.95,
        market_close_utc=asof_ts.replace(hour=21, minute=0),
        post_end_utc=asof_ts.replace(hour=1, minute=0) + timedelta(days=1),
        seconds_to_close=1800,
    )


def _scope(*, horizon_minutes: int, session: str, replay_mode: str) -> dict:
    return {
        "horizon_kind": "FIXED",
        "horizon_minutes": horizon_minutes,
        "session": session,
        "regime": "DEFAULT",
        "replay_mode": replay_mode,
        "scope_contract_version": "calibration_scope/v2",
    }



def _provenance(*, hash_value: str, evidence_ref: str) -> dict:
    return {
        "artifact_hash": hash_value,
        "trained_from_utc": "2026-01-01T00:00:00+00:00",
        "trained_to_utc": "2026-01-31T23:59:59+00:00",
        "valid_from_utc": "2026-02-01T00:00:00+00:00",
        "valid_to_utc": "2026-03-31T23:59:59+00:00",
        "evidence_ref": evidence_ref,
        "fit_sample_count": 2048,
        "provenance_contract_version": "calibration_provenance/v1",
    }



def _meta_json(
    *,
    horizon_minutes: int,
    session: str,
    replay_mode: str,
    artifact_version: str,
    artifact_hash: str,
    evidence_ref: str,
    ood_state: str = "IN_DISTRIBUTION",
    ood_reason: str | None = None,
    suppression_reason: str | None = None,
    calibrated_vector: dict | None = None,
) -> dict:
    scope = _scope(horizon_minutes=horizon_minutes, session=session, replay_mode=replay_mode)
    probability_contract = {
        "is_coherent": True,
        "ood_state": ood_state,
        "ood_reason": ood_reason,
        "suppression_reason": suppression_reason,
        "calibration_artifact_ref": {
            "artifact_name": "phase1_calibration",
            "artifact_version": artifact_version,
            "target_name": "intraday_direction_3class",
            "target_version": "target_v3",
            "artifact_hash": artifact_hash,
            "calibration_scope": scope,
            "artifact_provenance": _provenance(hash_value=artifact_hash, evidence_ref=evidence_ref),
        },
        "ood_contract_version": "ood/v3",
        "output_domain_contract_version": "output_domain/v2",
    }
    if calibrated_vector is not None:
        probability_contract["calibrated_probability_vector"] = calibrated_vector

    return {
        "prediction_contract": {
            "target_name": "intraday_direction_3class",
            "target_version": "target_v3",
            "label_version": "label_v5",
            "threshold_policy_version": "threshold_v6",
            "calibration_scope": scope,
        },
        "probability_contract": probability_contract,
        "calibration_selection": {
            "reason_code": "selected",
            "artifact_hash": artifact_hash,
            "calibration_scope": scope,
        },
        "ood_state": ood_state,
        "ood_reason": ood_reason,
        "ood_assessment": {
            "state": ood_state,
            "primary_reason": ood_reason,
            "contract_version": "ood/v3",
        },
        "replay_governance": {
            "requested_replay_mode": replay_mode,
            "prediction_replay_mode": replay_mode,
            "calibration_request_replay_mode": replay_mode,
            "calibration_artifact_scope_replay_mode": replay_mode,
            "calibration_selection_reason": "SELECTED",
        },
        "output_domain_contract_version": "output_domain/v2",
        "horizon_contract": {
            "decision_path_contract_version": "decision_path/v3",
        },
    }



def _insert_prediction(
    db: DbWriter,
    con: duckdb.DuckDBPyConnection,
    *,
    snapshot_id: str,
    horizon_minutes: int,
    replay_mode: str,
    session_scope: str,
    artifact_version: str,
    artifact_hash: str,
    evidence_ref: str,
    outcome_label: str,
    brier_score: float | None,
    log_loss: float | None,
    suppression_reason: str | None = None,
    ood_state: str = "IN_DISTRIBUTION",
    ood_reason: str | None = None,
    prob_up: float | None = None,
    prob_down: float | None = None,
    prob_flat: float | None = None,
) -> None:
    vector = None
    if suppression_reason is None:
        vector = {"UP": prob_up, "DOWN": prob_down, "FLAT": prob_flat}
    payload = {
        "snapshot_id": snapshot_id,
        "horizon_minutes": horizon_minutes,
        "horizon_kind": "FIXED",
        "horizon_seconds": horizon_minutes * 60,
        "start_price": 410.0,
        "bias": 0.12,
        "confidence": 0.74 if suppression_reason is None else 0.0,
        "prob_up": prob_up,
        "prob_down": prob_down,
        "prob_flat": prob_flat,
        "model_name": "bounded_additive_score",
        "model_version": "2.1.0",
        "decision_state": "LONG" if suppression_reason is None else "NO_SIGNAL",
        "risk_gate_status": "PASS" if suppression_reason is None else "BLOCKED",
        "data_quality_state": "OK" if suppression_reason is None else "DEGRADED",
        "confidence_state": "HIGH" if suppression_reason is None else "UNKNOWN",
        "replay_mode": replay_mode,
        "meta_json": _meta_json(
            horizon_minutes=horizon_minutes,
            session=session_scope,
            replay_mode=replay_mode,
            artifact_version=artifact_version,
            artifact_hash=artifact_hash,
            evidence_ref=evidence_ref,
            ood_state=ood_state,
            ood_reason=ood_reason,
            suppression_reason=suppression_reason,
            calibrated_vector=vector,
        ),
    }
    prediction_id = db.insert_prediction(con, payload)
    con.execute(
        """
        UPDATE predictions
        SET outcome_realized = TRUE,
            realized_at_utc = ?,
            outcome_label = ?,
            brier_score = ?,
            log_loss = ?,
            is_correct = ?
        WHERE prediction_id = ?
        """,
        [
            datetime(2026, 3, 5, 15, 30, tzinfo=UTC),
            outcome_label,
            brier_score,
            log_loss,
            (suppression_reason is None and outcome_label == "UP"),
            prediction_id,
        ],
    )



def _seed_fixture_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "scorecard_fixture.duckdb"
    db = DbWriter(str(db_path), str(tmp_path / "scorecard_fixture.lock"))
    con = duckdb.connect(str(db_path))
    db.ensure_schema(con)

    rth_snapshot = _seed_snapshot(
        db,
        con,
        ticker="MSFT",
        asof_ts=datetime(2026, 3, 4, 15, 30, tzinfo=UTC),
        session_label="REG",
    )
    ah_snapshot = _seed_snapshot(
        db,
        con,
        ticker="MSFT",
        asof_ts=datetime(2026, 3, 4, 23, 0, tzinfo=UTC),
        session_label="AFTERHOURS",
    )

    _insert_prediction(
        db,
        con,
        snapshot_id=rth_snapshot,
        horizon_minutes=15,
        replay_mode="LIVE_LIKE_OBSERVED",
        session_scope="RTH",
        artifact_version="cal_live_v1",
        artifact_hash="hash_live_v1",
        evidence_ref="evidence://live/v1",
        outcome_label="UP",
        brier_score=0.08,
        log_loss=0.24,
        prob_up=0.72,
        prob_down=0.14,
        prob_flat=0.14,
    )
    _insert_prediction(
        db,
        con,
        snapshot_id=ah_snapshot,
        horizon_minutes=60,
        replay_mode="RESEARCH_RESTATED",
        session_scope="AFTERHOURS",
        artifact_version="cal_restated_v2",
        artifact_hash="hash_restated_v2",
        evidence_ref="evidence://restated/v2",
        outcome_label="DOWN",
        brier_score=0.19,
        log_loss=0.51,
        prob_up=0.17,
        prob_down=0.61,
        prob_flat=0.22,
        ood_state="DEGRADED",
        ood_reason="FEATURE_COVERAGE_BELOW_TARGET",
    )
    _insert_prediction(
        db,
        con,
        snapshot_id=rth_snapshot,
        horizon_minutes=30,
        replay_mode="LIVE_LIKE_OBSERVED",
        session_scope="RTH",
        artifact_version="cal_live_v1",
        artifact_hash="hash_live_v1",
        evidence_ref="evidence://live/v1",
        outcome_label="UP",
        brier_score=None,
        log_loss=None,
        suppression_reason="OOD_REJECTION",
        ood_state="OUT_OF_DISTRIBUTION",
        ood_reason="FEATURE_BOUNDARY_VIOLATION",
        prob_up=None,
        prob_down=None,
        prob_flat=None,
    )

    con.close()
    return db_path



def test_calibration_scorecard_generates_deterministic_evidence_package(tmp_path: Path):
    db_path = _seed_fixture_db(tmp_path)
    out_dir = tmp_path / "scorecard_out"

    outputs = generate_calibration_scorecard(db_path, out_dir, prefix="fixture_scorecard")

    expected_files = {
        "json",
        "segments_csv",
        "reliability_csv",
        "accepted_vs_suppressed_csv",
        "artifact_inventory_csv",
        "markdown",
    }
    assert expected_files == set(outputs)
    for path in outputs.values():
        assert path.exists()
        assert path.read_text(encoding="utf-8")

    summary = json.loads(outputs["json"].read_text(encoding="utf-8"))
    assert summary["overall_summary"]["prediction_count"] == 3
    assert summary["overall_summary"]["population_counts"]["ACCEPTED"] == 2
    assert summary["overall_summary"]["population_counts"]["SUPPRESSED"] == 1
    assert sorted(summary["overall_summary"]["distinct_calibration_artifact_hashes"]) == [
        "hash_live_v1",
        "hash_restated_v2",
    ]

    segment_df = pd.read_csv(outputs["segments_csv"])
    assert set(segment_df["replay_mode"]) == {"LIVE_LIKE_OBSERVED", "RESEARCH_RESTATED"}
    assert set(segment_df["decision_population"]) == {"ACCEPTED", "SUPPRESSED"}

    reliability_df = pd.read_csv(outputs["reliability_csv"])
    assert set(reliability_df["decision_population"]) == {"ACCEPTED"}
    assert set(reliability_df["calibration_artifact_hashes"]) == {"hash_live_v1", "hash_restated_v2"}

    artifact_df = pd.read_csv(outputs["artifact_inventory_csv"])
    assert set(artifact_df["calibration_artifact_hash"]) == {"hash_live_v1", "hash_restated_v2"}
    assert set(artifact_df["calibration_evidence_ref"]) == {"evidence://live/v1", "evidence://restated/v2"}

    markdown = outputs["markdown"].read_text(encoding="utf-8")
    assert "# Calibration Scorecard" in markdown
    assert "hash_live_v1" in markdown
    assert "hash_restated_v2" in markdown

    first_pass = {name: path.read_text(encoding="utf-8") for name, path in outputs.items()}
    outputs_second = generate_calibration_scorecard(db_path, out_dir, prefix="fixture_scorecard")
    second_pass = {name: path.read_text(encoding="utf-8") for name, path in outputs_second.items()}
    assert first_pass == second_pass
