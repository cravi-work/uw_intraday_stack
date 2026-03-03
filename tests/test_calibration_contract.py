import pytest

from src.models import (
    CalibrationArtifactRef,
    DataQualityState,
    DecisionGate,
    PredictionTargetSpec,
    RiskGateStatus,
    SignalState,
    bounded_additive_score,
    build_calibration_artifact_ref,
    build_prediction_target_spec,
)


def _gate() -> DecisionGate:
    return DecisionGate(
        data_quality_state=DataQualityState.VALID,
        risk_gate_status=RiskGateStatus.PASS,
        decision_state=SignalState.NEUTRAL,
    )


def _target_spec() -> PredictionTargetSpec:
    cfg = {
        "target_spec": {
            "target_name": "intraday_direction_3class",
            "target_version": "calibration_contract_v1",
            "class_labels": ["UP", "DOWN", "FLAT"],
        }
    }
    return build_prediction_target_spec(cfg, horizon_kind="FIXED", horizon_minutes=15, flat_threshold_pct=0.001)


def test_derived_calibration_version_is_deterministic_when_not_declared():
    cfg = {
        "model_name": "bounded_additive_score",
        "calibration": {
            "bins": [0.0, 0.5, 1.0],
            "mapped": [0.05, 0.5, 0.95],
        },
    }
    target_spec = _target_spec()

    ref_a = build_calibration_artifact_ref(cfg, target_spec=target_spec)
    ref_b = build_calibration_artifact_ref(cfg, target_spec=target_spec)
    ref_changed = build_calibration_artifact_ref(
        {
            "model_name": "bounded_additive_score",
            "calibration": {
                "bins": [0.0, 0.5, 1.0],
                "mapped": [0.10, 0.5, 0.90],
            },
        },
        target_spec=target_spec,
    )

    assert ref_a is not None
    assert ref_b is not None
    assert ref_changed is not None
    assert ref_a.artifact_version == ref_b.artifact_version
    assert ref_a.artifact_version != ref_changed.artifact_version



def test_non_numeric_calibration_bins_are_rejected_at_builder_boundary():
    target_spec = _target_spec()
    cfg = {
        "calibration": {
            "bins": [0.0, "not-a-number", 1.0],
            "mapped": [0.1, 0.5, 0.9],
        }
    }

    ref = build_calibration_artifact_ref(cfg, target_spec=target_spec)

    assert ref is None



def test_duplicate_bins_are_suppressed_before_calibrated_probability_emission():
    target_spec = _target_spec()
    bad_ref = CalibrationArtifactRef(
        artifact_name="dup_bins",
        artifact_version="dup_v1",
        target_name=target_spec.target_name,
        target_version=target_spec.target_version,
        bins=(0.0, 0.0, 1.0),
        mapped=(0.05, 0.5, 0.95),
    )

    pred = bounded_additive_score(
        {"smart_whale_pressure": 0.8},
        data_quality_score=1.0,
        weights={"smart_whale_pressure": 1.0},
        gate=_gate(),
        target_spec=target_spec,
        calibration_artifact_ref=bad_ref,
    )

    assert bad_ref.is_valid() is False
    assert pred.probability_output.raw_probability_vector is not None
    assert pred.probability_output.calibrated_probability_vector is None
    assert pred.prob_up is None
    assert pred.prob_down is None
    assert pred.prob_flat is None
    assert pred.suppression_reason == "INVALID_CALIBRATION_ARTIFACT"
