import datetime as dt

from src.ingest_engine import generate_predictions
from src.models import ReplayMode, SessionState

ASOF_UTC = dt.datetime(2026, 3, 3, 15, 0, tzinfo=dt.timezone.utc)


def _artifact(*, version: str, replay_mode: str) -> dict:
    return {
        "artifact_name": "tri_class_calibration",
        "artifact_version": version,
        "target_name": "intraday_direction_3class",
        "target_version": "replay_scope_v1",
        "scope": {
            "horizon_kind": "FIXED",
            "horizon_minutes": 15,
            "session": "RTH",
            "regime": "DEFAULT",
            "replay_mode": replay_mode,
        },
        "bins": [0.0, 0.5, 1.0],
        "mapped": [0.05, 0.5, 0.95],
    }


def _cfg(artifacts: list[dict]) -> dict:
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
        "model": {
            "model_name": "bounded_additive_score",
            "model_version": "replay_scope_model_v1",
            "target_spec": {
                "target_name": "intraday_direction_3class",
                "target_version": "replay_scope_v1",
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
                "registry_version": "registry.replay.v1",
                "default_regime": "DEFAULT",
                "selection_policy": {"require_scope_match": True},
                "artifacts": artifacts,
            },
        },
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


def _run(cfg: dict, *, replay_mode=ReplayMode.LIVE_LIKE_OBSERVED) -> dict:
    predictions = generate_predictions(
        cfg,
        snapshot_id=321,
        valid_features=[_feature()],
        asof_utc=ASOF_UTC,
        session_enum=SessionState.RTH,
        sec_to_close=None,
        endpoint_coverage=1.0,
        replay_mode=replay_mode,
    )
    assert len(predictions) == 1
    return predictions[0]


def test_generate_predictions_selects_mode_specific_calibration_artifact():
    cfg = _cfg(
        [
            _artifact(version="cal.live_like", replay_mode="LIVE_LIKE_OBSERVED"),
            _artifact(version="cal.research", replay_mode="RESEARCH_RESTATED"),
        ]
    )

    live_like = _run(cfg, replay_mode=ReplayMode.LIVE_LIKE_OBSERVED)
    research = _run(cfg, replay_mode=ReplayMode.RESEARCH_RESTATED)

    assert live_like["replay_mode"] == "LIVE_LIKE_OBSERVED"
    assert research["replay_mode"] == "RESEARCH_RESTATED"

    live_selection = live_like["meta_json"]["calibration_selection"]
    research_selection = research["meta_json"]["calibration_selection"]

    assert live_selection["request"]["replay_mode"] == "LIVE_LIKE_OBSERVED"
    assert research_selection["request"]["replay_mode"] == "RESEARCH_RESTATED"
    assert live_selection["artifact"]["artifact_version"] == "cal.live_like"
    assert research_selection["artifact"]["artifact_version"] == "cal.research"

    assert live_like["meta_json"]["prediction_contract"]["calibration_request_replay_mode"] == "LIVE_LIKE_OBSERVED"
    assert research["meta_json"]["prediction_contract"]["calibration_request_replay_mode"] == "RESEARCH_RESTATED"
    assert live_like["prob_up"] is not None
    assert research["prob_up"] is not None


def test_generate_predictions_suppresses_when_replay_mode_scope_mismatches():
    cfg = _cfg([_artifact(version="cal.research", replay_mode="RESEARCH_RESTATED")])

    prediction = _run(cfg, replay_mode=ReplayMode.LIVE_LIKE_OBSERVED)
    selection = prediction["meta_json"]["calibration_selection"]

    assert selection["reason_code"] == "REPLAY_MODE_MISMATCH"
    assert prediction["meta_json"]["prediction_contract"]["calibration_selection_reason"] == "REPLAY_MODE_MISMATCH"
    assert prediction["meta_json"]["prediction_contract"]["calibration_version"] is None
    assert prediction["prob_up"] is None
    assert prediction["prob_down"] is None
    assert prediction["prob_flat"] is None
    assert prediction["meta_json"]["suppression_reason"] == "MISSING_CALIBRATION_ARTIFACT"


def test_generate_predictions_forward_path_defaults_to_live_like_observed():
    cfg = _cfg([_artifact(version="cal.live_like", replay_mode="LIVE_LIKE_OBSERVED")])

    predictions = generate_predictions(
        cfg,
        snapshot_id=322,
        valid_features=[_feature()],
        asof_utc=ASOF_UTC,
        session_enum=SessionState.RTH,
        sec_to_close=None,
        endpoint_coverage=1.0,
    )

    assert len(predictions) == 1
    prediction = predictions[0]
    selection = prediction["meta_json"]["calibration_selection"]

    assert prediction["replay_mode"] == "LIVE_LIKE_OBSERVED"
    assert prediction["meta_json"]["replay_mode"] == "LIVE_LIKE_OBSERVED"
    assert selection["request"]["replay_mode"] == "LIVE_LIKE_OBSERVED"
    assert selection["artifact"]["artifact_version"] == "cal.live_like"
