import datetime as dt
import logging

from src.calibration_registry import select_calibration_artifact
from src.ingest_engine import generate_predictions
from src.models import OODState, SessionState, build_prediction_target_spec


ASOF_UTC = dt.datetime(2026, 3, 3, 15, 0, tzinfo=dt.timezone.utc)


def _target_spec(model_cfg: dict, *, horizon_kind: str = "FIXED", horizon_minutes: int | None = 15):
    return build_prediction_target_spec(
        model_cfg,
        horizon_kind=horizon_kind,
        horizon_minutes=horizon_minutes,
        flat_threshold_pct=0.001,
    )


def _registry_artifact(*, version: str, horizon_kind: str, horizon_minutes: int | None = None, session: str = "ANY") -> dict:
    return {
        "artifact_name": "tri_class_calibration",
        "artifact_version": version,
        "target_name": "intraday_direction_3class",
        "target_version": "scope_test_v1",
        "scope": {
            "horizon_kind": horizon_kind,
            **({"horizon_minutes": horizon_minutes} if horizon_minutes is not None else {}),
            "session": session,
            "regime": "DEFAULT",
            "replay_mode": "ANY",
        },
        "bins": [0.0, 0.5, 1.0],
        "mapped": [0.05, 0.5, 0.95],
    }


def _model_cfg(artifacts: list[dict]) -> dict:
    return {
        "model_name": "bounded_additive_score",
        "model_version": "scope_test_model_v1",
        "target_spec": {
            "target_name": "intraday_direction_3class",
            "target_version": "scope_test_v1",
        },
        "confidence_cap": 0.55,
        "min_confidence": 0.35,
        "neutral_threshold": 0.55,
        "direction_margin": 0.08,
        "min_flat_prob": 0.15,
        "max_flat_prob": 0.65,
        "flat_from_data_quality_scale": 0.9,
        "weights": {"spot": 1.0, "oi_pressure": 0.5},
        "calibration_registry": {
            "registry_version": "registry.scope.v1",
            "default_regime": "DEFAULT",
            "selection_policy": {"require_scope_match": True},
            "artifacts": artifacts,
        },
    }


def _cfg(artifacts: list[dict]) -> dict:
    return {
        "ingestion": {"cadence_minutes": 5},
        "validation": {
            "horizon_weights_source": "explicit",
            "horizons_minutes": [15],
            "horizon_weights": {"15": {"spot": 1.0, "oi_pressure": 0.5}},
            "horizon_critical_features": {"15": ["spot", "oi_pressure"]},
            "use_default_required_features": False,
            "emit_to_close_horizon": False,
            "flat_threshold_pct": 0.001,
            "alignment_tolerance_sec": 900,
            "invalid_after_minutes": 60,
            "fallback_max_age_minutes": 15,
            "tolerance_minutes": 10,
            "max_horizon_drift_minutes": 10,
        },
        "model": _model_cfg(artifacts),
    }


def _feature(feature_key: str, value: float, *, session_applicability: str = "RTH") -> dict:
    return {
        "feature_key": feature_key,
        "feature_value": value,
        "meta_json": {
            "source_endpoints": [
                {
                    "method": "GET",
                    "path": f"/api/test/{feature_key}",
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
                "source_path": f"/api/test/{feature_key}",
                "units_expected": "Directional Imbalance Ratio [-1, 1]" if feature_key != "spot" else "Spot Price",
                "session_applicability": session_applicability,
                "decision_path_role": "signal-critical",
                "feature_use_contract_version": "feature_use/v1",
                "time_provenance_degraded": False,
            },
            "details": {},
        },
    }


def _run(cfg: dict, *, session: SessionState = SessionState.RTH) -> dict:
    predictions = generate_predictions(
        cfg,
        snapshot_id=123,
        valid_features=[_feature("spot", 150.0, session_applicability="PREMARKET/RTH/AFTERHOURS"), _feature("oi_pressure", 0.4)],
        asof_utc=ASOF_UTC,
        session_enum=session,
        sec_to_close=None,
        endpoint_coverage=1.0,
    )
    assert len(predictions) == 1
    return predictions[0]


def test_select_calibration_artifact_picks_exact_scope_match():
    model_cfg = _model_cfg(
        [
            _registry_artifact(version="cal.15.any", horizon_kind="FIXED", horizon_minutes=15, session="ANY"),
            _registry_artifact(version="cal.15.rth", horizon_kind="FIXED", horizon_minutes=15, session="RTH"),
        ]
    )
    target_spec = _target_spec(model_cfg, horizon_kind="FIXED", horizon_minutes=15)

    selection = select_calibration_artifact(
        model_cfg,
        target_spec=target_spec,
        horizon_kind="FIXED",
        horizon_minutes=15,
        session_state=SessionState.RTH,
        regime="DEFAULT",
    )

    assert selection.reason_code == "SELECTED"
    assert selection.artifact is not None
    assert selection.artifact.artifact_version == "cal.15.rth"
    assert selection.compatible_candidate_count == 2
    assert selection.artifact.calibration_scope["session"] == "RTH"



def test_select_calibration_artifact_reports_missing_artifact():
    model_cfg = _model_cfg([])
    target_spec = _target_spec(model_cfg)

    selection = select_calibration_artifact(
        model_cfg,
        target_spec=target_spec,
        horizon_kind="FIXED",
        horizon_minutes=15,
        session_state=SessionState.RTH,
        regime="DEFAULT",
    )

    assert selection.artifact is None
    assert selection.reason_code in {"NO_ARTIFACTS_CONFIGURED", "INVALID_REGISTRY_CONFIGURATION"}



def test_select_calibration_artifact_reports_target_mismatch():
    model_cfg = _model_cfg([_registry_artifact(version="cal.15.rth", horizon_kind="FIXED", horizon_minutes=15, session="RTH")])
    wrong_model_cfg = {
        **model_cfg,
        "target_spec": {"target_name": "intraday_direction_3class", "target_version": "other_target_v2"},
    }
    target_spec = _target_spec(wrong_model_cfg)

    selection = select_calibration_artifact(
        model_cfg,
        target_spec=target_spec,
        horizon_kind="FIXED",
        horizon_minutes=15,
        session_state=SessionState.RTH,
        regime="DEFAULT",
    )

    assert selection.artifact is None
    assert selection.reason_code == "TARGET_MISMATCH"



def test_select_calibration_artifact_reports_horizon_mismatch():
    model_cfg = _model_cfg([_registry_artifact(version="cal.15.rth", horizon_kind="FIXED", horizon_minutes=15, session="RTH")])
    target_spec = _target_spec(model_cfg, horizon_kind="FIXED", horizon_minutes=60)

    selection = select_calibration_artifact(
        model_cfg,
        target_spec=target_spec,
        horizon_kind="FIXED",
        horizon_minutes=60,
        session_state=SessionState.RTH,
        regime="DEFAULT",
    )

    assert selection.artifact is None
    assert selection.reason_code == "HORIZON_MISMATCH"



def test_select_calibration_artifact_reports_session_mismatch():
    model_cfg = _model_cfg([_registry_artifact(version="cal.15.rth", horizon_kind="FIXED", horizon_minutes=15, session="RTH")])
    target_spec = _target_spec(model_cfg)

    selection = select_calibration_artifact(
        model_cfg,
        target_spec=target_spec,
        horizon_kind="FIXED",
        horizon_minutes=15,
        session_state=SessionState.AFTERHOURS,
        regime="DEFAULT",
    )

    assert selection.artifact is None
    assert selection.reason_code == "SESSION_MISMATCH"



def test_generate_predictions_uses_scoped_calibration_artifact_and_logs_selection(caplog):
    cfg = _cfg([
        _registry_artifact(version="cal.15.rth", horizon_kind="FIXED", horizon_minutes=15, session="RTH"),
    ])

    with caplog.at_level(logging.INFO):
        prediction = _run(cfg, session=SessionState.RTH)

    contract = prediction["meta_json"]["prediction_contract"]
    selection = prediction["meta_json"]["calibration_selection"]

    assert prediction["prob_up"] is not None
    assert prediction["prob_down"] is not None
    assert prediction["prob_flat"] is not None
    assert contract["calibration_version"] == "cal.15.rth"
    assert contract["calibration_selection_reason"] == "SELECTED"
    assert selection["artifact"]["calibration_scope"]["session"] == "RTH"
    assert any("calibration_artifact_selected" in record.getMessage() for record in caplog.records)



def test_generate_predictions_suppresses_when_calibration_scope_mismatch(caplog):
    cfg = _cfg([
        _registry_artifact(version="cal.15.rth", horizon_kind="FIXED", horizon_minutes=15, session="RTH"),
    ])

    with caplog.at_level(logging.WARNING):
        prediction = _run(cfg, session=SessionState.AFTERHOURS)

    assert prediction["prob_up"] is None
    assert prediction["prob_down"] is None
    assert prediction["prob_flat"] is None
    assert prediction["meta_json"]["prediction_contract"]["calibration_selection_reason"] == "SESSION_MISMATCH"
    assert prediction["meta_json"]["calibration_selection"]["reason_code"] == "SESSION_MISMATCH"
    assert prediction["meta_json"]["suppression_reason"] == "MISSING_CALIBRATION_ARTIFACT"
    assert any("calibration_artifact_unavailable: SESSION_MISMATCH" in record.getMessage() for record in caplog.records)
