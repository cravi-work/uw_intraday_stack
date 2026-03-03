from datetime import datetime, timedelta, timezone

import duckdb
import pytest

from src.replay_engine import run_replay
from src.storage import DbWriter

UTC = timezone.utc


def _base_cfg(db_path: str) -> dict:
    return {
        "ingestion": {
            "watchlist": ["AAPL"],
            "cadence_minutes": 5,
            "enable_market_context": False,
            "premarket_start_et": "04:00",
            "regular_start_et": "09:30",
            "regular_end_et": "16:00",
            "afterhours_end_et": "20:00",
            "ingest_start_et": "04:00",
            "ingest_end_et": "20:00",
        },
        "storage": {
            "duckdb_path": db_path,
            "cycle_lock_path": db_path + ".cycle.lock",
            "writer_lock_path": db_path + ".writer.lock",
        },
        "system": {},
        "network": {},
        "validation": {
            "invalid_after_minutes": 60,
            "tolerance_minutes": 10,
            "max_horizon_drift_minutes": 10,
            "flat_threshold_pct": 0.001,
            "fallback_max_age_minutes": 15,
            "alignment_tolerance_sec": 900,
            "use_default_required_features": False,
            "emit_to_close_horizon": False,
            "horizon_weights_source": "explicit",
            "horizons_minutes": [15],
            "horizon_critical_features": {"15": ["spot"]},
            "horizon_weights": {"15": {"spot": 1.0}},
        },
        "model": {
            "model_name": "bounded_additive_score",
            "model_version": "model_v9",
            "target_spec": {
                "target_name": "intraday_direction_3class",
                "target_version": "target_v4",
                "class_labels": ["UP", "DOWN", "FLAT"],
                "probability_tolerance": 1e-6,
            },
            "label_contract": {
                "label_version": "label_v5",
                "session_boundary_rule": "TRUNCATE_TO_SESSION_CLOSE",
                "flat_threshold_policy": "ABS_RETURN_BAND",
                "threshold_policy_version": "threshold_v2",
            },
            "calibration": {
                "artifact_name": "bounded_additive_score_calibration",
                "artifact_version": "cal_v3",
                "bins": [0.0, 0.5, 1.0],
                "mapped": [0.1, 0.5, 0.9],
            },
        },
    }


def _insert_snapshot(db: DbWriter, con: duckdb.DuckDBPyConnection, *, asof: datetime, ticker: str = "AAPL") -> str:
    cfg_version = db.insert_config(con, "model: {}\n")
    run_id = db.begin_run(
        con,
        asof_ts_utc=asof,
        session_label="RTH",
        is_trading_day=True,
        is_early_close=False,
        config_version=cfg_version,
        api_catalog_hash="catalog_hash_v1",
    )
    return db.insert_snapshot(
        con,
        run_id=run_id,
        asof_ts_utc=asof,
        ticker=ticker,
        session_label="RTH",
        is_trading_day=True,
        is_early_close=False,
        data_quality_score=1.0,
        market_close_utc=asof + timedelta(hours=1),
        post_end_utc=asof + timedelta(hours=4),
        seconds_to_close=3600,
    )


def _bootstrap_observed_snapshot(
    db_path: str,
    *,
    restated_flag: bool = False,
    future_publish: bool = False,
    received_after_snapshot: bool = False,
    source_revision: str = "rev-1",
) -> datetime:
    db = DbWriter(db_path)
    asof = datetime(2026, 1, 3, 15, 0, tzinfo=UTC)
    with db.writer() as con:
        db.ensure_schema(con)
        snapshot_id = _insert_snapshot(db, con, asof=asof)
        con.execute(
            "INSERT INTO dim_endpoints (endpoint_id, method, path, signature, params_hash, params_json) VALUES (1, 'GET', '/api/stock/{ticker}/ohlc/{candle_size}', 'GET /api/stock/{ticker}/ohlc/1m', NULL, '{}')"
        )

        payload_ts = asof - timedelta(minutes=1)
        publish_ts = asof + timedelta(minutes=2) if future_publish else payload_ts
        received_ts = asof + timedelta(minutes=1) if received_after_snapshot else payload_ts
        payload = [{"close": 150.0, "t": int(payload_ts.timestamp())}]
        event_id = db.insert_raw_event(
            con,
            run_id=None,
            ticker="AAPL",
            endpoint_id=1,
            req_at=payload_ts,
            rec_at=received_ts,
            status=200,
            lat=10,
            ph="hash-1",
            pj=payload,
            retry=False,
            etype=None,
            emsg=None,
            circ=None,
            source_publish_time_utc=publish_ts,
            source_revision=source_revision,
        )
        details = {
            "effective_ts_utc": payload_ts.isoformat(),
            "endpoint_asof_ts_utc": asof.isoformat(),
            "truth_status": "SUCCESS_HAS_DATA",
            "stale_age_seconds": 60,
            "source_publish_time_utc": publish_ts.isoformat(),
            "source_revision": source_revision,
        }
        if restated_flag:
            details["replay_source_mode"] = "RESEARCH_RESTATED"
        db.insert_lineage(
            con,
            snapshot_id=snapshot_id,
            endpoint_id=1,
            used_event_id=event_id,
            freshness_state="FRESH",
            data_age_seconds=60,
            payload_class="SUCCESS_HAS_DATA",
            na_reason=None,
            meta_json={
                "source_endpoints": [],
                "freshness_state": "FRESH",
                "stale_age_min": 1,
                "na_reason": None,
                "details": details,
            },
        )
    return asof


def _prediction_meta(*, replay_mode: str = "UNKNOWN", calibration_version: str = "cal_v3") -> dict:
    return {
        "prediction_contract": {
            "target_name": "intraday_direction_3class",
            "target_version": "target_v4",
            "label_version": "label_v5",
            "threshold_policy_version": "threshold_v2",
            "target_spec": {
                "target_name": "intraday_direction_3class",
                "target_version": "target_v4",
            },
            "label_contract": {
                "label_version": "label_v5",
                "threshold_policy_version": "threshold_v2",
            },
        },
        "probability_contract": {
            "target_spec": {
                "target_name": "intraday_direction_3class",
                "target_version": "target_v4",
            },
            "calibration_artifact_ref": {
                "artifact_name": "bounded_additive_score_calibration",
                "artifact_version": calibration_version,
                "target_name": "intraday_direction_3class",
                "target_version": "target_v4",
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
        "replay_mode": replay_mode,
        "ood_state": "UNKNOWN",
    }


def _insert_prediction(
    db: DbWriter,
    con: duckdb.DuckDBPyConnection,
    *,
    snapshot_id: str,
    replay_mode: str,
    calibration_version: str,
) -> None:
    db.insert_prediction(
        con,
        {
            "snapshot_id": snapshot_id,
            "horizon_minutes": 15,
            "horizon_kind": "FIXED",
            "horizon_seconds": None,
            "start_price": 150.0,
            "bias": 0.25,
            "confidence": 0.5,
            "prob_up": 0.6,
            "prob_down": 0.2,
            "prob_flat": 0.2,
            "model_name": "bounded_additive_score",
            "model_version": "model_v9",
            "model_hash": "hash-v1",
            "meta_json": _prediction_meta(replay_mode=replay_mode, calibration_version=calibration_version),
            "decision_state": "LONG",
            "risk_gate_status": "PASS",
            "confidence_state": "MEDIUM",
            "data_quality_state": "VALID",
            "blocked_reasons": [],
            "degraded_reasons": [],
            "validation_eligible": True,
            "gate_json": {"validation_eligible": True},
            "alignment_status": "ALIGNED",
            "source_ts_min_utc": datetime(2026, 1, 3, 14, 59, tzinfo=UTC),
            "source_ts_max_utc": datetime(2026, 1, 3, 15, 0, tzinfo=UTC),
            "critical_missing_count": 0,
            "decision_window_id": f"window-{replay_mode}-{calibration_version}",
            "replay_mode": replay_mode,
        },
    )



def test_live_like_replay_rejects_restated_inputs(tmp_path):
    db_path = str(tmp_path / "restated.duckdb")
    _bootstrap_observed_snapshot(db_path, restated_flag=True)

    with pytest.raises(RuntimeError, match="LIVE_LIKE_OBSERVED replay contaminated by restated/backfilled data"):
        run_replay(db_path, "AAPL", cfg=_base_cfg(db_path), replay_mode="LIVE_LIKE_OBSERVED")

    con = duckdb.connect(db_path)
    row = con.execute("SELECT replay_mode, status, failure_reason FROM replay_runs ORDER BY started_at_utc DESC LIMIT 1").fetchone()
    con.close()
    assert row[0] == "LIVE_LIKE_OBSERVED"
    assert row[1] == "FAILED"
    assert "restated/backfilled" in row[2]



def test_research_replay_persists_mode_and_frozen_versions(tmp_path):
    db_path = str(tmp_path / "research.duckdb")
    _bootstrap_observed_snapshot(db_path)

    report = run_replay(db_path, "AAPL", cfg=_base_cfg(db_path), replay_mode="RESEARCH_RESTATED")

    assert report["status"] == "PASSED"
    assert report["replay_mode"] == "RESEARCH_RESTATED"
    assert report["snapshot_count"] == 1
    assert report["prediction_count"] == 1
    assert report["frozen_contract"]["model_version"] == "model_v9"
    assert report["frozen_contract"]["calibration_version"] == "cal_v3"
    assert report["frozen_contract"]["threshold_policy_version"] == "threshold_v2"
    assert report["frozen_contract"]["target_version"] == "target_v4"
    assert report["frozen_contract"]["label_version"] == "label_v5"
    assert report["frozen_contract"]["feature_version"].startswith("derived_feature_contract_")

    pred = report["recomputed_predictions"][0]
    assert pred["replay_mode"] == "RESEARCH_RESTATED"
    assert pred["meta_json"]["replay_mode"] == "RESEARCH_RESTATED"
    assert pred["meta_json"]["replay_contract"]["model_version"] == "model_v9"
    assert pred["meta_json"]["replay_contract"]["calibration_version"] == "cal_v3"

    con = duckdb.connect(db_path)
    row = con.execute(
        """
        SELECT replay_mode, model_version, feature_version, calibration_version,
               threshold_policy_version, target_version, label_version,
               snapshot_count, prediction_count, status
        FROM replay_runs
        ORDER BY started_at_utc DESC
        LIMIT 1
        """
    ).fetchone()
    con.close()

    assert row[0] == "RESEARCH_RESTATED"
    assert row[1] == "model_v9"
    assert row[2].startswith("derived_feature_contract_")
    assert row[3] == "cal_v3"
    assert row[4] == "threshold_v2"
    assert row[5] == "target_v4"
    assert row[6] == "label_v5"
    assert row[7] == 1
    assert row[8] == 1
    assert row[9] == "PASSED"



def test_replay_rejects_mixed_replay_modes(tmp_path):
    db_path = str(tmp_path / "mixed_modes.duckdb")
    db = DbWriter(db_path)
    asof = datetime(2026, 1, 3, 15, 0, tzinfo=UTC)
    with db.writer() as con:
        db.ensure_schema(con)
        snapshot_a = _insert_snapshot(db, con, asof=asof)
        snapshot_b = _insert_snapshot(db, con, asof=asof + timedelta(minutes=5))
        _insert_prediction(db, con, snapshot_id=snapshot_a, replay_mode="LIVE_LIKE_OBSERVED", calibration_version="cal_v3")
        _insert_prediction(db, con, snapshot_id=snapshot_b, replay_mode="RESEARCH_RESTATED", calibration_version="cal_v3")

    with pytest.raises(RuntimeError, match="Mixed replay modes are invalid"):
        run_replay(db_path, "AAPL", cfg=_base_cfg(db_path), replay_mode="LIVE_LIKE_OBSERVED")



def test_replay_rejects_artifact_version_drift(tmp_path):
    db_path = str(tmp_path / "artifact_drift.duckdb")
    db = DbWriter(db_path)
    asof = datetime(2026, 1, 3, 15, 0, tzinfo=UTC)
    with db.writer() as con:
        db.ensure_schema(con)
        snapshot_a = _insert_snapshot(db, con, asof=asof)
        snapshot_b = _insert_snapshot(db, con, asof=asof + timedelta(minutes=5))
        _insert_prediction(db, con, snapshot_id=snapshot_a, replay_mode="UNKNOWN", calibration_version="cal_v3")
        _insert_prediction(db, con, snapshot_id=snapshot_b, replay_mode="UNKNOWN", calibration_version="cal_v99")

    with pytest.raises(RuntimeError, match="Artifact version drift inside replay run: calibration_version"):
        run_replay(db_path, "AAPL", cfg=_base_cfg(db_path), replay_mode="RESEARCH_RESTATED")



def test_live_like_replay_rejects_future_publish_time_contamination(tmp_path):
    db_path = str(tmp_path / "future_publish.duckdb")
    _bootstrap_observed_snapshot(db_path, future_publish=True)

    with pytest.raises(RuntimeError, match="LIVE_LIKE_OBSERVED replay contaminated by restated/backfilled data"):
        run_replay(db_path, "AAPL", cfg=_base_cfg(db_path), replay_mode="LIVE_LIKE_OBSERVED")

    con = duckdb.connect(db_path)
    row = con.execute("SELECT failure_reason FROM replay_runs ORDER BY started_at_utc DESC LIMIT 1").fetchone()
    con.close()
    assert "source_publish_after_snapshot" in row[0]



def test_live_like_replay_rejects_received_after_snapshot_contamination(tmp_path):
    db_path = str(tmp_path / "future_received.duckdb")
    _bootstrap_observed_snapshot(db_path, received_after_snapshot=True)

    with pytest.raises(RuntimeError, match="LIVE_LIKE_OBSERVED replay contaminated by restated/backfilled data"):
        run_replay(db_path, "AAPL", cfg=_base_cfg(db_path), replay_mode="LIVE_LIKE_OBSERVED")

    con = duckdb.connect(db_path)
    row = con.execute("SELECT failure_reason FROM replay_runs ORDER BY started_at_utc DESC LIMIT 1").fetchone()
    con.close()
    assert "received_after_snapshot" in row[0]



def test_live_like_replay_rejects_revision_marker_contamination(tmp_path):
    db_path = str(tmp_path / "revision_marker.duckdb")
    _bootstrap_observed_snapshot(db_path, source_revision="backfilled_rev_2")

    with pytest.raises(RuntimeError, match="LIVE_LIKE_OBSERVED replay contaminated by restated/backfilled data"):
        run_replay(db_path, "AAPL", cfg=_base_cfg(db_path), replay_mode="LIVE_LIKE_OBSERVED")

    con = duckdb.connect(db_path)
    row = con.execute("SELECT failure_reason FROM replay_runs ORDER BY started_at_utc DESC LIMIT 1").fetchone()
    con.close()
    assert "revision_marks_restated" in row[0]
