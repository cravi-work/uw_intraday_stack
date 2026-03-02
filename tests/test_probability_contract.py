import pytest

from src.endpoint_truth import EndpointContext
from src.features import extract_darkpool_pressure, extract_litflow_pressure, extract_oi_features
from src.models import (
    DataQualityState,
    DecisionGate,
    RiskGateStatus,
    SignalState,
    bounded_additive_score,
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


def test_adversarial_unsigned_totals_no_long_saturation():
    ctx = _ctx()
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
    )

    assert huge_oi is None
    assert huge_darkpool is None
    assert balanced_litflow == pytest.approx(0.0)
    assert pred.bias == pytest.approx(0.0)
    assert pred.gate.decision_state == SignalState.NEUTRAL
    assert pred.prob_up == pred.prob_down
