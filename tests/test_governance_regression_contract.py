import datetime as dt

import pytest

import src.ingest_engine as ie_mod
from src.models import OODState, SessionState
from src.ood import OODAssessment


ASOF_UTC = dt.datetime(2026, 3, 3, 15, 0, tzinfo=dt.timezone.utc)


def _base_cfg(*, weights=None, criticals=None, calibration_artifacts=None):
    weights = dict(weights or {"oi_pressure": 1.0})
    criticals = list(criticals or ["spot"])
    model = {
        "model_name": "bounded_additive_score",
        "model_version": "governance.test.v1",
        "confidence_cap": 0.55,
        "min_confidence": 0.35,
        "neutral_threshold": 0.55,
        "direction_margin": 0.08,
        "min_flat_prob": 0.15,
        "max_flat_prob": 0.65,
        "flat_from_data_quality_scale": 0.9,
        "target_spec": {
            "target_name": "intraday_direction_3class",
            "target_version": "governance.target.v1",
        },
    }
    if calibration_artifacts is None:
        model["calibration_registry"] = {
            "contract_version": "calibration_registry/v1",
            "registry_version": "governance.registry.v1",
            "default_regime": "DEFAULT",
            "selection_policy": {
                "require_scope_match": True,
                "allow_legacy_fallback": False,
            },
            "compatibility_rules": {
                "require_target_match": True,
                "require_horizon_match": True,
                "require_session_match": True,
                "require_regime_match": True,
                "require_replay_mode_match": False,
            },
            "artifacts": [
                {
                    "artifact_name": "governance_calibration",
                    "artifact_version": "cal.rth.5.v1",
                    "target_name": "intraday_direction_3class",
                    "target_version": "governance.target.v1",
                    "bins": [0.0, 0.5, 1.0],
                    "mapped": [0.05, 0.5, 0.95],
                    "scope": {
                        "horizon_kind": "FIXED",
                        "horizon_minutes": 5,
                        "session": "RTH",
                        "regime": "DEFAULT",
                        "scope_contract_version": "calibration_scope/v1",
                    },
                }
            ],
        }
    else:
        model["calibration_registry"] = calibration_artifacts

    return {
        "ingestion": {"cadence_minutes": 5},
        "validation": {
            "horizon_weights_source": "explicit",
            "horizons_minutes": [5],
            "horizon_weights": {"5": weights},
            "horizon_critical_features": {"5": criticals},
            "use_default_required_features": False,
            "emit_to_close_horizon": False,
            "flat_threshold_pct": 0.001,
            "alignment_tolerance_sec": 900,
            "invalid_after_minutes": 60,
            "decision_path_policy": {
                "contract_version": "decision_path/v1",
                "zero_weight_is_non_decision": True,
                "require_feature_metadata": True,
                "allow_explicit_zero_weight_critical_override": True,
                "explicit_zero_weight_critical_features": {},
            },
        },
        "model": {
            **model,
            "ood_assessment_policy": {
                "contract_version": "ood_assessment/v1",
                "require_assessment_before_emission": True,
                "degraded_coverage_threshold": 0.85,
                "out_coverage_threshold": 0.50,
                "boundary_slack": 1e-6,
            },
            "ood_probability_policy": {
                "contract_version": "ood_probability/v1",
                "degraded_confidence_scale": 0.6,
                "degraded_suppress_calibrated_probability": False,
                "unknown_suppress_calibrated_probability": True,
                "out_of_distribution_suppress_calibrated_probability": True,
                "unknown_confidence_scale": 0.0,
                "out_of_distribution_confidence_scale": 0.0,
            },
        },
    }



def _feature(
    feature_key,
    value,
    *,
    path,
    use_role="signal-critical",
    decision_eligible=None,
    units_expected=None,
    session_applicability="PREMARKET/RTH/AFTERHOURS",
):
    if decision_eligible is None:
        decision_eligible = use_role == "signal-critical"
    contract = {
        "contract_version": "feature_use/v1",
        "use_role": use_role,
        "decision_path": bool(decision_eligible),
        "decision_eligible": bool(decision_eligible),
        "missing_affects_confidence": bool(decision_eligible),
        "stale_affects_confidence": bool(decision_eligible),
    }
    return {
        "feature_key": feature_key,
        "feature_value": value,
        "meta_json": {
            "source_endpoints": [
                {
                    "method": "GET",
                    "path": path,
                    "purpose": use_role,
                    "decision_path": bool(decision_eligible),
                    "missing_affects_confidence": bool(decision_eligible),
                    "stale_affects_confidence": bool(decision_eligible),
                    "purpose_contract_version": "endpoint_purpose/v1",
                }
            ],
            "freshness_state": "FRESH",
            "stale_age_min": 0,
            "feature_use_contract": contract,
            "use_role": use_role,
            "decision_eligible": bool(decision_eligible),
            "missing_affects_confidence": bool(decision_eligible),
            "stale_affects_confidence": bool(decision_eligible),
            "metric_lineage": {
                "effective_ts_utc": (ASOF_UTC - dt.timedelta(minutes=1)).isoformat(),
                "source_path": path,
                "units_expected": units_expected or ("Spot Price" if feature_key == "spot" else "Directional Imbalance Ratio [-1, 1]"),
                "session_applicability": session_applicability,
                "decision_path_role": use_role,
                "feature_use_contract_version": "feature_use/v1",
                "time_provenance_degraded": False,
            },
            "details": {},
        },
    }



def _run(cfg, features, *, session=SessionState.RTH):
    predictions = ie_mod.generate_predictions(
        cfg,
        snapshot_id=999,
        valid_features=features,
        asof_utc=ASOF_UTC,
        session_enum=session,
        sec_to_close=None,
        endpoint_coverage=1.0,
    )
    assert len(predictions) == 1
    return predictions[0]



def test_forward_path_report_only_zero_weight_feature_is_fully_irrelevant():
    cfg = _base_cfg(weights={"oi_pressure": 1.0, "darkpool_pressure": 0.0}, criticals=["spot"])
    base = [
        _feature("spot", 150.0, path="/api/stock/{ticker}/ohlc/{candle_size}"),
        _feature("oi_pressure", 0.8, path="/api/stock/{ticker}/oi-per-strike"),
    ]
    report_only = _feature(
        "darkpool_pressure",
        0.45,
        path="/api/darkpool/{ticker}",
        use_role="report-only",
        decision_eligible=False,
    )

    pred_without = _run(cfg, base)
    pred_with = _run(cfg, base + [report_only])

    assert pred_without["decision_state"] == pred_with["decision_state"]
    assert pred_without["confidence"] == pred_with["confidence"]
    assert pred_without["prob_up"] == pred_with["prob_up"]
    assert pred_without["prob_down"] == pred_with["prob_down"]
    assert pred_without["prob_flat"] == pred_with["prob_flat"]
    assert pred_without["meta_json"]["decision_dq"] == pred_with["meta_json"]["decision_dq"] == 1.0
    assert pred_with["meta_json"]["horizon_contract"]["report_only_excluded_features"] == ["darkpool_pressure"]
    assert pred_with["meta_json"]["horizon_contract"]["zero_weight_excluded_features"] == ["darkpool_pressure"]



def test_forward_path_context_only_feature_missing_value_does_not_change_confidence():
    cfg = _base_cfg(weights={"oi_pressure": 1.0, "vol_term_slope": 1.0}, criticals=["spot"])
    base = [
        _feature("spot", 150.0, path="/api/stock/{ticker}/ohlc/{candle_size}"),
        _feature("oi_pressure", 0.6, path="/api/stock/{ticker}/oi-per-strike"),
    ]
    context_only_present = _feature(
        "vol_term_slope",
        0.2,
        path="/api/market/market-tide",
        use_role="context-only",
        decision_eligible=False,
    )
    context_only_missing = _feature(
        "vol_term_slope",
        None,
        path="/api/market/market-tide",
        use_role="context-only",
        decision_eligible=False,
    )

    pred_valid = _run(cfg, base + [context_only_present])
    pred_missing = _run(cfg, base + [context_only_missing])

    assert pred_valid["confidence"] == pred_missing["confidence"]
    assert pred_valid["decision_state"] == pred_missing["decision_state"]
    assert pred_valid["data_quality_state"] == pred_missing["data_quality_state"] == "VALID"
    assert pred_missing["meta_json"]["horizon_contract"]["context_only_excluded_features"] == ["vol_term_slope"]
    assert pred_missing["meta_json"]["dq_reason_codes"] == []



def test_forward_path_unknown_ood_suppresses_probability_output(monkeypatch):
    cfg = _base_cfg()
    features = [
        _feature("spot", 150.0, path="/api/stock/{ticker}/ohlc/{candle_size}"),
        _feature("oi_pressure", 0.4, path="/api/stock/{ticker}/oi-per-strike"),
    ]

    def _fake_unknown_ood(**kwargs):
        return OODAssessment(
            state=OODState.UNKNOWN,
            primary_reason="assessment_unavailable",
            reasons=("assessment_unavailable",),
            decision_feature_keys=("oi_pressure",),
            valid_decision_feature_count=1,
            total_decision_feature_count=1,
            coverage_ratio=1.0,
            session_state="RTH",
            assessment_ran=False,
        )

    monkeypatch.setattr(ie_mod, "assess_operational_ood", _fake_unknown_ood)
    pred = _run(cfg, features)

    assert pred["prob_up"] is None
    assert pred["prob_down"] is None
    assert pred["prob_flat"] is None
    assert pred["confidence_state"] == "UNKNOWN"
    assert pred["meta_json"]["ood_state"] == "UNKNOWN"
    assert pred["meta_json"]["ood_reason"] == "assessment_unavailable"
    assert pred["meta_json"]["suppression_reason"] == "OOD_UNKNOWN"



def test_forward_path_calibration_scope_mismatch_suppresses_calibrated_probability():
    cfg = _base_cfg()
    features = [
        _feature("spot", 150.0, path="/api/stock/{ticker}/ohlc/{candle_size}"),
        _feature("oi_pressure", 0.4, path="/api/stock/{ticker}/oi-per-strike"),
    ]

    pred = _run(cfg, features, session=SessionState.AFTERHOURS)

    assert pred["prob_up"] is None
    assert pred["prob_down"] is None
    assert pred["prob_flat"] is None
    assert pred["meta_json"]["calibration_selection"]["reason_code"] == "SESSION_MISMATCH"
    assert pred["meta_json"]["prediction_contract"]["calibration_version"] is None
    assert pred["meta_json"]["suppression_reason"] == "MISSING_CALIBRATION_ARTIFACT"
