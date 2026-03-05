
import datetime as dt
import logging

from src.ingest_engine import generate_predictions
from src.models import OODState, SessionState
from src.ood import assess_operational_ood


ASOF_UTC = dt.datetime(2026, 3, 3, 15, 0, tzinfo=dt.timezone.utc)


def _cfg(weights: dict[str, float], criticals: list[str]) -> dict:
    return {
        "ingestion": {"cadence_minutes": 5},
        "validation": {
            "horizon_weights_source": "explicit",
            "horizons_minutes": [5],
            "horizon_weights": {"5": dict(weights)},
            "horizon_critical_features": {"5": list(criticals)},
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
            "model_version": "test.v1",
            "confidence_cap": 0.55,
            "min_confidence": 0.35,
            "neutral_threshold": 0.55,
            "direction_margin": 0.08,
            "min_flat_prob": 0.15,
            "max_flat_prob": 0.65,
            "flat_from_data_quality_scale": 0.9,
            "target_spec": {
                "target_name": "intraday_direction_3class",
                "target_version": "test.v1",
            },
            "calibration": {
                "artifact_name": "test_calibration",
                "artifact_version": "test.cal.v1",
                "bins": [0.0, 0.5, 1.0],
                "mapped": [0.05, 0.5, 0.95],
            },
        },
    }


def _feature(
    feature_key: str,
    value: float | None,
    *,
    path: str,
    units_expected: str,
    session_applicability: str,
    time_provenance_degraded: bool = False,
    na_reason: str | None = None,
    use_role: str = "signal-critical",
    decision_eligible: bool = True,
    bounded_output: bool = False,
    expected_bounds: tuple[float, float] | dict | None = None,
    emitted_units: str | None = None,
    raw_input_units: str | None = None,
    output_domain: str | None = None,
    output_domain_contract_version: str | None = None,
    allowed_values: list[float] | None = None,
) -> dict:
    contract = {
        "contract_version": "feature_use/v1",
        "use_role": use_role,
        "decision_path": bool(decision_eligible),
        "decision_eligible": bool(decision_eligible),
        "missing_affects_confidence": bool(decision_eligible),
        "stale_affects_confidence": bool(decision_eligible),
    }
    lineage = {
        "effective_ts_utc": (ASOF_UTC - dt.timedelta(minutes=1)).isoformat(),
        "source_path": path,
        "units_expected": units_expected,
        "session_applicability": session_applicability,
        "decision_path_role": use_role,
        "feature_use_contract_version": "feature_use/v1",
        "time_provenance_degraded": bool(time_provenance_degraded),
    }
    if bounded_output:
        lineage.update(
            {
                "bounded_output": True,
                "expected_bounds": (
                    {"lower": float(expected_bounds[0]), "upper": float(expected_bounds[1]), "inclusive": True}
                    if isinstance(expected_bounds, tuple)
                    else expected_bounds
                ),
                "emitted_units": emitted_units,
                "raw_input_units": raw_input_units,
                "output_domain": output_domain,
                "output_domain_contract_version": output_domain_contract_version,
                "allowed_values": list(allowed_values) if allowed_values is not None else None,
            }
        )
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
                    "purpose_contract_version": "feature_use/v1",
                }
            ],
            "freshness_state": "FRESH",
            "stale_age_min": 0,
            "na_reason": na_reason,
            "feature_use_contract": contract,
            "use_role": use_role,
            "decision_eligible": bool(decision_eligible),
            "missing_affects_confidence": bool(decision_eligible),
            "stale_affects_confidence": bool(decision_eligible),
            "metric_lineage": lineage,
            "details": {},
        },
    }


def _run(features: list[dict], cfg: dict, *, session: SessionState = SessionState.RTH) -> dict:
    predictions = generate_predictions(
        cfg,
        snapshot_id=123,
        valid_features=features,
        asof_utc=ASOF_UTC,
        session_enum=session,
        sec_to_close=None,
        endpoint_coverage=1.0,
    )
    assert len(predictions) == 1
    return predictions[0]


def test_assess_operational_ood_marks_healthy_bundle_in_distribution():
    features = [
        _feature(
            "spot",
            150.0,
            path="/api/stock/{ticker}/ohlc/{candle_size}",
            units_expected="Spot Price",
            session_applicability="PREMARKET/RTH/AFTERHOURS",
        ),
        _feature(
            "oi_pressure",
            0.45,
            path="/api/stock/{ticker}/oi-per-strike",
            units_expected="Directional Imbalance Ratio [-1, 1]",
            session_applicability="RTH",
            bounded_output=True,
            expected_bounds=(-1.0, 1.0),
            emitted_units="directional_imbalance_ratio",
            raw_input_units="Open Interest (contracts)",
            output_domain="closed_interval",
            output_domain_contract_version="output_domain/v1",
        ),
    ]

    assessment = assess_operational_ood(
        feature_rows=features,
        decision_feature_keys=["spot", "oi_pressure"],
        session_state=SessionState.RTH,
    )

    assert assessment.state == OODState.IN_DISTRIBUTION
    assert assessment.primary_reason == "decision_feature_bundle_in_distribution"
    assert assessment.coverage_ratio == 1.0
    assert assessment.boundary_violation_features == {}


def test_assess_operational_ood_marks_time_provenance_degradation():
    features = [
        _feature(
            "spot",
            150.0,
            path="/api/stock/{ticker}/ohlc/{candle_size}",
            units_expected="Spot Price",
            session_applicability="PREMARKET/RTH/AFTERHOURS",
        ),
        _feature(
            "oi_pressure",
            0.40,
            path="/api/stock/{ticker}/oi-per-strike",
            units_expected="Directional Imbalance Ratio [-1, 1]",
            session_applicability="RTH",
            time_provenance_degraded=True,
            bounded_output=True,
            expected_bounds=(-1.0, 1.0),
            emitted_units="directional_imbalance_ratio",
            raw_input_units="Open Interest (contracts)",
            output_domain="closed_interval",
            output_domain_contract_version="output_domain/v1",
        ),
    ]

    assessment = assess_operational_ood(
        feature_rows=features,
        decision_feature_keys=["spot", "oi_pressure"],
        session_state=SessionState.RTH,
    )

    assert assessment.state == OODState.DEGRADED
    assert assessment.primary_reason.startswith("time_provenance_degraded:")
    assert assessment.degraded_feature_keys == ("oi_pressure",)


def test_assess_operational_ood_marks_boundary_violation_out_of_distribution():
    features = [
        _feature(
            "spot",
            150.0,
            path="/api/stock/{ticker}/ohlc/{candle_size}",
            units_expected="Spot Price",
            session_applicability="PREMARKET/RTH/AFTERHOURS",
        ),
        _feature(
            "oi_pressure",
            1.40,
            path="/api/stock/{ticker}/oi-per-strike",
            units_expected="Directional Imbalance Ratio [-1, 1]",
            session_applicability="RTH",
            bounded_output=True,
            expected_bounds=(-1.0, 1.0),
            emitted_units="directional_imbalance_ratio",
            raw_input_units="Open Interest (contracts)",
            output_domain="closed_interval",
            output_domain_contract_version="output_domain/v1",
        ),
    ]

    assessment = assess_operational_ood(
        feature_rows=features,
        decision_feature_keys=["spot", "oi_pressure"],
        session_state=SessionState.RTH,
    )

    assert assessment.state == OODState.OUT_OF_DISTRIBUTION
    assert assessment.primary_reason == "boundary_violation:oi_pressure"
    assert "oi_pressure" in assessment.boundary_violation_features


def test_assess_operational_ood_returns_unknown_when_assessment_cannot_run():
    assessment = assess_operational_ood(
        feature_rows=[],
        decision_feature_keys=[],
        session_state=SessionState.RTH,
    )

    assert assessment.state == OODState.UNKNOWN
    assert assessment.primary_reason == "no_decision_features_configured"
    assert assessment.assessment_ran is False


def test_generate_predictions_emits_real_ood_state_and_logs_boundary_violation(caplog):
    cfg = _cfg(weights={"oi_pressure": 1.0}, criticals=["spot"])
    features = [
        _feature(
            "spot",
            150.0,
            path="/api/stock/{ticker}/ohlc/{candle_size}",
            units_expected="Spot Price",
            session_applicability="PREMARKET/RTH/AFTERHOURS",
        ),
        _feature(
            "oi_pressure",
            1.25,
            path="/api/stock/{ticker}/oi-per-strike",
            units_expected="Directional Imbalance Ratio [-1, 1]",
            session_applicability="RTH",
            bounded_output=True,
            expected_bounds=(-1.0, 1.0),
            emitted_units="directional_imbalance_ratio",
            raw_input_units="Open Interest (contracts)",
            output_domain="closed_interval",
            output_domain_contract_version="output_domain/v1",
        ),
    ]

    with caplog.at_level(logging.INFO):
        pred = _run(features, cfg)

    assert pred["meta_json"]["ood_state"] == OODState.OUT_OF_DISTRIBUTION.value
    assert pred["meta_json"]["ood_reason"] == "boundary_violation:oi_pressure"
    assert pred["meta_json"]["ood_assessment"]["boundary_violation_features"]["oi_pressure"]["upper_bound"] == 1.0
    assert pred["prob_up"] is None
    counters = [getattr(record, "counter", None) for record in caplog.records]
    assert "ood_rejection_count" in counters
    assert "ood_feature_boundary_violation_count" in counters


def test_generate_predictions_marks_degraded_for_time_provenance_degraded_bundle(caplog):
    cfg = _cfg(weights={"oi_pressure": 1.0}, criticals=["spot"])
    features = [
        _feature(
            "spot",
            150.0,
            path="/api/stock/{ticker}/ohlc/{candle_size}",
            units_expected="Spot Price",
            session_applicability="PREMARKET/RTH/AFTERHOURS",
        ),
        _feature(
            "oi_pressure",
            0.35,
            path="/api/stock/{ticker}/oi-per-strike",
            units_expected="Directional Imbalance Ratio [-1, 1]",
            session_applicability="RTH",
            time_provenance_degraded=True,
            bounded_output=True,
            expected_bounds=(-1.0, 1.0),
            emitted_units="directional_imbalance_ratio",
            raw_input_units="Open Interest (contracts)",
            output_domain="closed_interval",
            output_domain_contract_version="output_domain/v1",
        ),
    ]

    with caplog.at_level(logging.INFO):
        pred = _run(features, cfg)

    assert pred["meta_json"]["ood_state"] == OODState.DEGRADED.value
    assert pred["meta_json"]["ood_reason"].startswith("time_provenance_degraded:")
    counters = [getattr(record, "counter", None) for record in caplog.records]
    assert "ood_degraded_count" in counters
