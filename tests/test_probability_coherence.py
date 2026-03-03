import pytest

from src.models import (
    CalibrationArtifactRef,
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
        target_version="coherence_v1",
    )


def _cal_ref(*, target_version: str = "coherence_v1") -> CalibrationArtifactRef:
    return CalibrationArtifactRef(
        artifact_name="tri_class_calibration",
        artifact_version="coh_cal_v1",
        target_name="intraday_direction_3class",
        target_version=target_version,
        bins=(0.0, 0.5, 1.0),
        mapped=(0.02, 0.5, 0.98),
    )


def test_calibrated_probability_vector_is_coherent():
    pred = bounded_additive_score(
        {"smart_whale_pressure": 0.9, "net_gex_sign": 0.4},
        data_quality_score=1.0,
        weights={"smart_whale_pressure": 0.8, "net_gex_sign": 0.2},
        gate=_gate(),
        model_name="bounded_additive_score",
        model_version="3.0.0",
        target_spec=_target_spec(),
        calibration_artifact_ref=_cal_ref(),
    )

    assert pred.prob_up is not None
    assert pred.prob_down is not None
    assert pred.prob_flat is not None
    assert 0.0 <= pred.prob_up <= 1.0
    assert 0.0 <= pred.prob_down <= 1.0
    assert 0.0 <= pred.prob_flat <= 1.0
    assert pred.prob_up + pred.prob_down + pred.prob_flat == pytest.approx(1.0, abs=1e-4)
    assert pred.probability_output.is_coherent()


def test_calibration_target_mismatch_suppresses_probability_output():
    pred = bounded_additive_score(
        {"smart_whale_pressure": -0.7},
        data_quality_score=1.0,
        weights={"smart_whale_pressure": 1.0},
        gate=_gate(),
        target_spec=_target_spec(),
        calibration_artifact_ref=_cal_ref(target_version="other_target_v2"),
    )

    assert pred.prob_up is None
    assert pred.prob_down is None
    assert pred.prob_flat is None
    assert pred.suppression_reason == "CALIBRATION_TARGET_MISMATCH"


def test_ood_rejection_suppresses_probability_output():
    pred = bounded_additive_score(
        {"smart_whale_pressure": 0.7},
        data_quality_score=1.0,
        weights={"smart_whale_pressure": 1.0},
        gate=_gate(),
        target_spec=_target_spec(),
        calibration_artifact_ref=_cal_ref(),
        ood_state=OODState.OUT_OF_DISTRIBUTION,
    )

    assert pred.prob_up is None
    assert pred.prob_down is None
    assert pred.prob_flat is None
    assert pred.suppression_reason == "OOD_REJECTION"
    assert pred.probability_output.calibrated_probability_vector is None



def test_extreme_positive_bias_calibration_remains_coherent():
    pred = bounded_additive_score(
        {"smart_whale_pressure": 1.0, "net_gex_sign": 1.0},
        data_quality_score=1.0,
        weights={"smart_whale_pressure": 0.7, "net_gex_sign": 0.3},
        gate=_gate(),
        target_spec=_target_spec(),
        calibration_artifact_ref=_cal_ref(),
    )

    assert pred.prob_up is not None
    assert pred.prob_down is not None
    assert pred.prob_flat is not None
    assert pred.prob_up > pred.prob_down
    assert pred.prob_up + pred.prob_down + pred.prob_flat == pytest.approx(1.0, abs=1e-4)
    assert pred.probability_output.is_coherent()



def test_extreme_negative_bias_calibration_remains_coherent():
    pred = bounded_additive_score(
        {"smart_whale_pressure": -1.0, "net_gex_sign": -1.0},
        data_quality_score=1.0,
        weights={"smart_whale_pressure": 0.7, "net_gex_sign": 0.3},
        gate=_gate(),
        target_spec=_target_spec(),
        calibration_artifact_ref=_cal_ref(),
    )

    assert pred.prob_up is not None
    assert pred.prob_down is not None
    assert pred.prob_flat is not None
    assert pred.prob_down > pred.prob_up
    assert pred.prob_up + pred.prob_down + pred.prob_flat == pytest.approx(1.0, abs=1e-4)
    assert pred.probability_output.is_coherent()
