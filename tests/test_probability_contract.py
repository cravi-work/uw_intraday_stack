import pytest

from src.endpoint_truth import EndpointContext
from src.features import extract_darkpool_pressure, extract_litflow_pressure, extract_oi_features
from src.models import (
    CalibrationArtifactRef,
    ConfidenceState,
    DataQualityState,
    DecisionGate,
    OODState,
    PredictionTargetSpec,
    ReplayMode,
    build_label_contract_spec,
    RiskGateStatus,
    SignalState,
    bounded_additive_score,
    build_calibration_artifact_ref,
    build_prediction_target_spec,
)


def _ctx() -> EndpointContext:
    return EndpointContext(
        endpoint_id=1,
        method="GET",
        path="/api/test",
        operation_id="op",
        signature="GET /api/test",
        used_event_id="evt-1",
        payload_class="SUCCESS_HAS_DATA",
        freshness_state="FRESH",
        stale_age_min=0,
        na_reason=None,
        effective_time_source="missing_provider_time",
        timestamp_quality="DEGRADED",
        time_provenance_degraded=True,
    )


def _contract():
    model_cfg = {
        "model_name": "bounded_additive_score",
        "model_version": "2.0.0",
        "target_spec": {
            "target_name": "intraday_direction_3class",
            "target_version": "test_target_v1",
            "class_labels": ["UP", "DOWN", "FLAT"],
            "probability_tolerance": 1e-6,
        },
        "calibration": {
            "artifact_name": "bounded_additive_score_calibration",
            "artifact_version": "cal_v1",
            "bins": [0.0, 0.5, 1.0],
            "mapped": [0.05, 0.5, 0.95],
        },
    }
    target_spec = build_prediction_target_spec(
        model_cfg,
        horizon_kind="FIXED",
        horizon_minutes=15,
        flat_threshold_pct=0.001,
    )
    cal_ref = build_calibration_artifact_ref(model_cfg, target_spec=target_spec)
    return model_cfg, target_spec, cal_ref


def test_adversarial_unsigned_totals_no_long_saturation():
    ctx = _ctx()
    _, target_spec, cal_ref = _contract()

    huge_oi = extract_oi_features(
        [
            {"strike": 100.0, "open_interest": 9.0e12},
            {"strike": 105.0, "open_interest": 8.5e12},
        ],
        ctx,
    ).features["oi_pressure"]
    huge_darkpool = extract_darkpool_pressure(
        [
            {"price": 1000.0, "volume": 9.0e9},
            {"price": 999.0, "size": 8.0e9},
        ],
        ctx,
    ).features["darkpool_pressure"]
    balanced_litflow = extract_litflow_pressure(
        [
            {"price": 1000.0, "volume": 9.0e9, "side": "BUY"},
            {"price": 1000.0, "volume": 9.0e9, "side": "SELL"},
        ],
        ctx,
    ).features["litflow_pressure"]

    pred = bounded_additive_score(
        {
            "oi_pressure": huge_oi,
            "darkpool_pressure": huge_darkpool,
            "litflow_pressure": balanced_litflow,
        },
        data_quality_score=1.0,
        weights={"oi_pressure": 0.30, "darkpool_pressure": 0.0, "litflow_pressure": 0.25},
        gate=DecisionGate(
            data_quality_state=DataQualityState.VALID,
            risk_gate_status=RiskGateStatus.PASS,
            decision_state=SignalState.NEUTRAL,
        ),
        target_spec=target_spec,
        calibration_artifact_ref=cal_ref,
        ood_state=OODState.IN_DISTRIBUTION,
    )

    assert huge_oi is None
    assert huge_darkpool is None
    assert balanced_litflow == pytest.approx(0.0)
    assert pred.bias == pytest.approx(0.0)
    assert pred.gate.decision_state == SignalState.NEUTRAL
    assert pred.prob_up == pred.prob_down
    assert pred.probability_output.is_coherent()


def test_probability_output_suppressed_without_target_spec():
    pred = bounded_additive_score(
        {"smart_whale_pressure": 0.6},
        data_quality_score=1.0,
        weights={"smart_whale_pressure": 1.0},
        gate=DecisionGate(
            data_quality_state=DataQualityState.VALID,
            risk_gate_status=RiskGateStatus.PASS,
            decision_state=SignalState.NEUTRAL,
        ),
        ood_state=OODState.IN_DISTRIBUTION,
    )

    assert pred.prob_up is None
    assert pred.prob_down is None
    assert pred.prob_flat is None
    assert pred.suppression_reason == "MISSING_TARGET_SPEC"
    assert pred.confidence_state == ConfidenceState.DEGRADED
    assert pred.probability_output.calibrated_probability_vector is None


def test_missing_calibration_suppresses_probabilities_and_degrades_confidence():
    _, target_spec, _ = _contract()

    pred = bounded_additive_score(
        {"smart_whale_pressure": 0.6},
        data_quality_score=1.0,
        weights={"smart_whale_pressure": 1.0},
        gate=DecisionGate(
            data_quality_state=DataQualityState.VALID,
            risk_gate_status=RiskGateStatus.PASS,
            decision_state=SignalState.NEUTRAL,
        ),
        target_spec=target_spec,
        ood_state=OODState.IN_DISTRIBUTION,
    )

    assert pred.prob_up is None
    assert pred.prob_down is None
    assert pred.prob_flat is None
    assert pred.suppression_reason == "MISSING_CALIBRATION_ARTIFACT"
    assert pred.confidence_state == ConfidenceState.DEGRADED


def test_model_and_contract_versions_propagate():
    model_cfg, target_spec, cal_ref = _contract()

    pred = bounded_additive_score(
        {"smart_whale_pressure": 0.8, "net_gex_sign": 0.4},
        data_quality_score=1.0,
        weights={"smart_whale_pressure": 0.7, "net_gex_sign": 0.3},
        gate=DecisionGate(
            data_quality_state=DataQualityState.VALID,
            risk_gate_status=RiskGateStatus.PASS,
            decision_state=SignalState.NEUTRAL,
        ),
        model_name=model_cfg["model_name"],
        model_version=model_cfg["model_version"],
        target_spec=target_spec,
        calibration_artifact_ref=cal_ref,
        replay_mode=ReplayMode.UNKNOWN,
        ood_state=OODState.IN_DISTRIBUTION,
    )

    assert pred.model_name == "bounded_additive_score"
    assert pred.model_version == "2.0.0"
    assert pred.probability_output.target_spec.target_name == "intraday_direction_3class"
    assert pred.probability_output.target_spec.target_version == "test_target_v1"
    assert pred.probability_output.calibration_artifact_ref.artifact_name == "bounded_additive_score_calibration"
    assert pred.probability_output.calibration_artifact_ref.artifact_version == "cal_v1"
    assert pred.probability_output.is_coherent()


def test_invalid_calibration_artifact_suppresses_output():
    target_spec = PredictionTargetSpec(target_name="intraday_direction_3class", target_version="v1")
    bad_cal_ref = CalibrationArtifactRef(
        artifact_name="bad",
        artifact_version="v1",
        target_name="intraday_direction_3class",
        target_version="v1",
        bins=(0.0, 0.5, 0.5),
        mapped=(0.0, 0.5, 1.0),
    )

    pred = bounded_additive_score(
        {"smart_whale_pressure": 0.4},
        data_quality_score=1.0,
        weights={"smart_whale_pressure": 1.0},
        gate=DecisionGate(
            data_quality_state=DataQualityState.VALID,
            risk_gate_status=RiskGateStatus.PASS,
            decision_state=SignalState.NEUTRAL,
        ),
        target_spec=target_spec,
        calibration_artifact_ref=bad_cal_ref,
        ood_state=OODState.IN_DISTRIBUTION,
    )

    assert pred.suppression_reason == "INVALID_CALIBRATION_ARTIFACT"
    assert pred.probability_output.calibrated_probability_vector is None


def test_label_contract_builder_persists_versions_and_threshold_policy():
    model_cfg = {
        "neutral_threshold": 0.55,
        "direction_margin": 0.08,
    }
    validation_cfg = {
        "flat_threshold_pct": 0.001,
        "label_contract": {
            "label_version": "label_v2",
            "threshold_policy_version": "thresholds_v2",
            "session_boundary_rule": "TRUNCATE_TO_SESSION_CLOSE",
        },
    }

    label_contract = build_label_contract_spec(model_cfg, validation_cfg, flat_threshold_pct=0.001)

    assert label_contract.is_valid()
    assert label_contract.label_version == "label_v2"
    assert label_contract.threshold_policy_version == "thresholds_v2"
    assert label_contract.neutral_threshold == pytest.approx(0.55)
    assert label_contract.direction_margin == pytest.approx(0.08)



def test_missing_calibration_preserves_raw_vector_but_suppresses_calibrated_output():
    _, target_spec, _ = _contract()

    pred = bounded_additive_score(
        {"smart_whale_pressure": 0.65, "net_gex_sign": 0.20},
        data_quality_score=1.0,
        weights={"smart_whale_pressure": 0.8, "net_gex_sign": 0.2},
        gate=DecisionGate(
            data_quality_state=DataQualityState.VALID,
            risk_gate_status=RiskGateStatus.PASS,
            decision_state=SignalState.NEUTRAL,
        ),
        target_spec=target_spec,
        ood_state=OODState.IN_DISTRIBUTION,
    )

    assert pred.raw_score > 0.0
    assert pred.probability_output.raw_probability_vector is not None
    assert pred.probability_output.calibrated_probability_vector is None
    assert pred.prob_up is None
    assert pred.prob_down is None
    assert pred.prob_flat is None
    assert pred.suppression_reason == "MISSING_CALIBRATION_ARTIFACT"



def test_invalid_target_spec_suppresses_probability_output_before_emission():
    bad_target_spec = PredictionTargetSpec(
        target_name="intraday_direction_3class",
        target_version="",
    )
    _, _, cal_ref = _contract()

    pred = bounded_additive_score(
        {"smart_whale_pressure": 0.75},
        data_quality_score=1.0,
        weights={"smart_whale_pressure": 1.0},
        gate=DecisionGate(
            data_quality_state=DataQualityState.VALID,
            risk_gate_status=RiskGateStatus.PASS,
            decision_state=SignalState.NEUTRAL,
        ),
        target_spec=bad_target_spec,
        calibration_artifact_ref=cal_ref,
        ood_state=OODState.IN_DISTRIBUTION,
    )

    assert pred.raw_score > 0.0
    assert pred.probability_output.raw_probability_vector is not None
    assert pred.probability_output.calibrated_probability_vector is None
    assert pred.prob_up is None
    assert pred.prob_down is None
    assert pred.prob_flat is None
    assert pred.suppression_reason == "INVALID_TARGET_SPEC"
