import pytest

from src.models import (
    CalibrationArtifactRef,
    ConfidenceState,
    DataQualityState,
    DecisionGate,
    OODState,
    PredictionTargetSpec,
    RiskGateStatus,
    SignalState,
    bounded_additive_score,
)



def _gate() -> DecisionGate:
    return DecisionGate(
        data_quality_state=DataQualityState.VALID,
        risk_gate_status=RiskGateStatus.PASS,
        decision_state=SignalState.NEUTRAL,
    )



def _target_spec() -> PredictionTargetSpec:
    return PredictionTargetSpec(
        target_name="intraday_direction_3class",
        target_version="probability_ood_v1",
    )



def _calibration_ref() -> CalibrationArtifactRef:
    return CalibrationArtifactRef(
        artifact_name="probability_ood_calibration",
        artifact_version="probability_ood_cal_v1",
        target_name="intraday_direction_3class",
        target_version="probability_ood_v1",
        bins=(0.0, 0.5, 1.0),
        mapped=(0.05, 0.5, 0.95),
    )



def test_degraded_ood_lowers_confidence_and_marks_contract_degraded():
    kwargs = {
        "features": {"smart_whale_pressure": 0.9, "net_gex_sign": 0.3},
        "data_quality_score": 1.0,
        "weights": {"smart_whale_pressure": 0.8, "net_gex_sign": 0.2},
        "gate": _gate(),
        "target_spec": _target_spec(),
        "calibration_artifact_ref": _calibration_ref(),
    }

    in_distribution = bounded_additive_score(
        **kwargs,
        ood_state=OODState.IN_DISTRIBUTION,
        ood_reason="decision_feature_bundle_in_distribution",
    )
    degraded = bounded_additive_score(
        **kwargs,
        ood_state=OODState.DEGRADED,
        ood_reason="time_provenance_degraded:oi_pressure",
    )

    assert in_distribution.prob_up is not None
    assert degraded.prob_up is not None
    assert degraded.confidence < in_distribution.confidence
    assert degraded.confidence_state == ConfidenceState.DEGRADED
    assert degraded.probability_output.ood_reason == "time_provenance_degraded:oi_pressure"
    assert degraded.probability_output.ood_policy_action == "degrade_confidence"



def test_unknown_ood_suppresses_calibrated_output_and_sets_unknown_confidence():
    pred = bounded_additive_score(
        {"smart_whale_pressure": 0.7},
        data_quality_score=1.0,
        weights={"smart_whale_pressure": 1.0},
        gate=_gate(),
        target_spec=_target_spec(),
        calibration_artifact_ref=_calibration_ref(),
        ood_state=OODState.UNKNOWN,
        ood_reason="assessment_unavailable",
    )

    assert pred.probability_output.raw_probability_vector is not None
    assert pred.probability_output.calibrated_probability_vector is None
    assert pred.prob_up is None
    assert pred.prob_down is None
    assert pred.prob_flat is None
    assert pred.suppression_reason == "OOD_UNKNOWN"
    assert pred.confidence == pytest.approx(0.0)
    assert pred.confidence_state == ConfidenceState.UNKNOWN
    assert pred.probability_output.ood_policy_action == "suppress"
    assert pred.probability_output.ood_reason == "assessment_unavailable"
