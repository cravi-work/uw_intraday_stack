import datetime as dt
import json

from src.calibration_registry import select_calibration_artifact
from src.ingest_engine import generate_predictions
from src.models import CalibrationArtifactRef, ReplayMode, SessionState, build_prediction_target_spec
from src.storage import DbWriter

ASOF_UTC = dt.datetime(2026, 3, 3, 15, 0, tzinfo=dt.timezone.utc)
UTC = dt.timezone.utc


def _artifact(*, version: str, replay_mode: str = "ANY", include_provenance: bool = True, evidence_ref: str = "evidence://calibration/report-v1") -> dict:
    artifact = {
        "artifact_name": "tri_class_calibration",
        "artifact_version": version,
        "target_name": "intraday_direction_3class",
        "target_version": "registry_contract_v1",
        "scope": {
            "horizon_kind": "FIXED",
            "horizon_minutes": 15,
            "session": "RTH",
            "regime": "DEFAULT",
            "replay_mode": replay_mode,
            "scope_contract_version": "calibration_scope/v1",
        },
        "bins": [0.0, 0.5, 1.0],
        "mapped": [0.05, 0.5, 0.95],
    }
    if include_provenance:
        artifact["provenance"] = {
            "provenance_contract_version": "calibration_provenance/v1",
            "trained_from_utc": "2026-01-01T00:00:00+00:00",
            "trained_to_utc": "2026-01-31T23:59:59+00:00",
            "valid_from_utc": "2026-02-01T00:00:00+00:00",
            "valid_to_utc": "2026-03-31T23:59:59+00:00",
            "evidence_ref": evidence_ref,
            "fit_sample_count": 2048,
        }
    return artifact


def _model_cfg(
    artifacts: list[dict],
    *,
    required_provenance_fields=None,
    allow_generic_scope_fallback: bool = True,
) -> dict:
    compatibility_rules = {
        "require_target_match": True,
        "require_horizon_match": True,
        "require_session_match": True,
        "require_regime_match": True,
        "require_replay_mode_match": True,
        "require_artifact_hash": True,
        "allow_generic_scope_fallback": allow_generic_scope_fallback,
    }
    if required_provenance_fields is not None:
        compatibility_rules["required_provenance_fields"] = list(required_provenance_fields)
    return {
        "model_name": "bounded_additive_score",
        "model_version": "registry_contract_model_v1",
        "target_spec": {
            "target_name": "intraday_direction_3class",
            "target_version": "registry_contract_v1",
        },
        "confidence_cap": 0.55,
        "min_confidence": 0.35,
        "neutral_threshold": 0.55,
        "direction_margin": 0.08,
        "min_flat_prob": 0.15,
        "max_flat_prob": 0.65,
        "flat_from_data_quality_scale": 0.9,
        "weights": {"spot": 1.0},
        "calibration_registry": {
            "registry_version": "registry.contract.v1",
            "default_regime": "DEFAULT",
            "selection_policy": {
                "require_scope_match": True,
                "allow_generic_scope_fallback": allow_generic_scope_fallback,
            },
            "compatibility_rules": compatibility_rules,
            "artifacts": artifacts,
        },
    }


def _cfg(artifacts: list[dict], **kwargs) -> dict:
    return {
        "ingestion": {"cadence_minutes": 5},
        "validation": {
            "horizon_weights_source": "explicit",
            "horizons_minutes": [15],
            "horizon_weights": {"15": {"spot": 1.0}},
            "horizon_critical_features": {"15": ["spot"]},
            "use_default_required_features": False,
            "emit_to_close_horizon": False,
            "flat_threshold_pct": 0.001,
            "alignment_tolerance_sec": 900,
            "invalid_after_minutes": 60,
            "fallback_max_age_minutes": 15,
            "tolerance_minutes": 10,
            "max_horizon_drift_minutes": 10,
        },
        "model": _model_cfg(artifacts, **kwargs),
    }


def _feature() -> dict:
    return {
        "feature_key": "spot",
        "feature_value": 150.0,
        "meta_json": {
            "source_endpoints": [
                {
                    "method": "GET",
                    "path": "/api/test/spot",
                    "purpose": "signal-critical",
                    "decision_path": True,
                    "missing_affects_confidence": True,
                    "stale_affects_confidence": True,
                    "purpose_contract_version": "feature_use/v1",
                }
            ],
            "freshness_state": "FRESH",
            "stale_age_min": 0,
            "feature_use_contract": {
                "contract_version": "feature_use/v1",
                "use_role": "signal-critical",
                "decision_path": True,
                "decision_eligible": True,
                "missing_affects_confidence": True,
                "stale_affects_confidence": True,
            },
            "use_role": "signal-critical",
            "decision_eligible": True,
            "missing_affects_confidence": True,
            "stale_affects_confidence": True,
            "metric_lineage": {
                "effective_ts_utc": (ASOF_UTC - dt.timedelta(minutes=1)).isoformat(),
                "source_path": "/api/test/spot",
                "units_expected": "Spot Price",
                "emitted_units": "Spot Price",
                "raw_input_units": "Spot Price",
                "bounded_output": False,
                "output_domain_contract_version": "feature_domain/v1",
                "session_applicability": "PREMARKET/RTH/AFTERHOURS",
                "decision_path_role": "signal-critical",
                "feature_use_contract_version": "feature_use/v1",
                "time_provenance_degraded": False,
            },
            "details": {},
        },
    }


def _bootstrap_snapshot(db: DbWriter):
    now = dt.datetime(2026, 2, 3, 15, 30, tzinfo=UTC)
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
            api_catalog_hash="catalog_hash_v3",
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
    return snapshot_id


def test_select_calibration_artifact_rejects_missing_required_provenance_under_policy():
    model_cfg = _model_cfg(
        [_artifact(version="cal.no_provenance", include_provenance=False)],
        required_provenance_fields=(
            "trained_from_utc",
            "trained_to_utc",
            "valid_from_utc",
            "valid_to_utc",
            "evidence_ref",
            "fit_sample_count",
        ),
    )
    selection = select_calibration_artifact(
        model_cfg,
        target_spec=build_prediction_target_spec(
            model_cfg,
            horizon_kind="FIXED",
            horizon_minutes=15,
            flat_threshold_pct=0.001,
        ),
        horizon_kind="FIXED",
        horizon_minutes=15,
        session_state=SessionState.RTH,
        regime="DEFAULT",
        replay_mode=ReplayMode.LIVE_LIKE_OBSERVED,
    )

    assert selection.artifact is None
    assert selection.reason_code == "MISSING_REQUIRED_PROVENANCE"
    assert set(selection.provenance_required_fields) == {
        "trained_from_utc",
        "trained_to_utc",
        "valid_from_utc",
        "valid_to_utc",
        "evidence_ref",
        "fit_sample_count",
    }
    assert any("trained_from_utc" in reason for reason in selection.reasons)
    assert any("evidence_ref" in reason for reason in selection.reasons)


def test_select_calibration_artifact_reports_replay_mode_mismatch_even_with_complete_provenance():
    model_cfg = _model_cfg([_artifact(version="cal.restated.only", replay_mode="RESEARCH_RESTATED")])
    target_spec = build_prediction_target_spec(
        model_cfg,
        horizon_kind="FIXED",
        horizon_minutes=15,
        flat_threshold_pct=0.001,
    )
    selection = select_calibration_artifact(
        model_cfg,
        target_spec=target_spec,
        horizon_kind="FIXED",
        horizon_minutes=15,
        session_state=SessionState.RTH,
        regime="DEFAULT",
        replay_mode=ReplayMode.LIVE_LIKE_OBSERVED,
    )

    assert selection.artifact is None
    assert selection.reason_code == "REPLAY_MODE_MISMATCH"


def test_artifact_hash_changes_when_provenance_changes():
    ref_a = CalibrationArtifactRef(
        artifact_name="tri_class_calibration",
        artifact_version="cal.v1",
        target_name="intraday_direction_3class",
        target_version="registry_contract_v1",
        bins=(0.0, 0.5, 1.0),
        mapped=(0.05, 0.5, 0.95),
        scope_horizon_kind="FIXED",
        scope_horizon_minutes=15,
        scope_session="RTH",
        scope_regime="DEFAULT",
        scope_replay_mode="ANY",
        trained_from_utc="2026-01-01T00:00:00+00:00",
        trained_to_utc="2026-01-31T23:59:59+00:00",
        valid_from_utc="2026-02-01T00:00:00+00:00",
        valid_to_utc="2026-03-31T23:59:59+00:00",
        evidence_ref="evidence://calibration/report-a",
        fit_sample_count=2048,
    )
    ref_b = CalibrationArtifactRef(
        artifact_name="tri_class_calibration",
        artifact_version="cal.v1",
        target_name="intraday_direction_3class",
        target_version="registry_contract_v1",
        bins=(0.0, 0.5, 1.0),
        mapped=(0.05, 0.5, 0.95),
        scope_horizon_kind="FIXED",
        scope_horizon_minutes=15,
        scope_session="RTH",
        scope_regime="DEFAULT",
        scope_replay_mode="ANY",
        trained_from_utc="2026-01-01T00:00:00+00:00",
        trained_to_utc="2026-01-31T23:59:59+00:00",
        valid_from_utc="2026-02-01T00:00:00+00:00",
        valid_to_utc="2026-03-31T23:59:59+00:00",
        evidence_ref="evidence://calibration/report-b",
        fit_sample_count=2048,
    )

    assert ref_a.is_valid() is True
    assert ref_b.is_valid() is True
    assert ref_a.artifact_hash != ref_b.artifact_hash


def test_selected_artifact_identity_is_persisted_and_selectable(tmp_path):
    db = DbWriter(str(tmp_path / "registry_contract.duckdb"), str(tmp_path / "registry_contract.lock"))
    snapshot_id = _bootstrap_snapshot(db)
    cfg = _cfg([
        _artifact(version="cal.persisted.v1", replay_mode="LIVE_LIKE_OBSERVED", evidence_ref="evidence://calibration/persisted"),
    ])

    predictions = generate_predictions(
        cfg,
        snapshot_id=snapshot_id,
        valid_features=[_feature()],
        asof_utc=ASOF_UTC,
        session_enum=SessionState.RTH,
        sec_to_close=None,
        endpoint_coverage=1.0,
        replay_mode=ReplayMode.LIVE_LIKE_OBSERVED,
    )
    assert len(predictions) == 1
    payload = predictions[0]

    with db.writer() as con:
        prediction_id = db.insert_prediction(con, payload)
        row = con.execute(
            """
            SELECT calibration_artifact_hash, calibration_scope, probability_contract_json
            FROM predictions WHERE prediction_id = ?
            """,
            [prediction_id],
        ).fetchone()

    contract = json.loads(row[2]) if isinstance(row[2], str) else row[2]
    stored_scope = json.loads(row[1]) if isinstance(row[1], str) else row[1]
    artifact_ref = contract["calibration_artifact_ref"]

    assert row[0] == artifact_ref["artifact_hash"]
    assert stored_scope["replay_mode"] == "LIVE_LIKE_OBSERVED"
    assert artifact_ref["artifact_provenance"]["evidence_ref"] == "evidence://calibration/persisted"
    assert artifact_ref["artifact_provenance"]["fit_sample_count"] == 2048
