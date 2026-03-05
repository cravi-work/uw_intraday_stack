
import datetime as dt
import logging

from src.ingest_engine import generate_predictions
from src.models import OODState, SessionState
from src.ood import assess_operational_ood

ASOF_UTC = dt.datetime(2026, 3, 3, 15, 0, tzinfo=dt.timezone.utc)


def _cfg() -> dict:
    return {
        "ingestion": {"cadence_minutes": 5},
        "validation": {
            "horizon_weights_source": "explicit",
            "horizons_minutes": [5],
            "horizon_weights": {"5": {"oi_pressure": 1.0}},
            "horizon_critical_features": {"5": ["spot"]},
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
    bounded_output: bool = False,
    expected_bounds: tuple[float, float] | dict | None = None,
    emitted_units: str | None = None,
    raw_input_units: str | None = None,
    output_domain: str | None = None,
    output_domain_contract_version: str | None = None,
    allowed_values: list[float] | None = None,
    units_expected: str = "legacy prose only",
) -> dict:
    lineage = {
        "effective_ts_utc": (ASOF_UTC - dt.timedelta(minutes=1)).isoformat(),
        "source_path": path,
        "units_expected": units_expected,
        "session_applicability": "RTH" if feature_key != "spot" else "PREMARKET/RTH/AFTERHOURS",
        "decision_path_role": "signal-critical",
        "feature_use_contract_version": "feature_use/v1",
        "time_provenance_degraded": False,
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
            "metric_lineage": lineage,
            "details": {},
        },
    }


def _run(features: list[dict], *, caplog=None) -> dict:
    cfg = _cfg()
    if caplog is None:
        predictions = generate_predictions(
            cfg,
            snapshot_id=123,
            valid_features=features,
            asof_utc=ASOF_UTC,
            session_enum=SessionState.RTH,
            sec_to_close=None,
            endpoint_coverage=1.0,
        )
    else:
        with caplog.at_level(logging.INFO):
            predictions = generate_predictions(
                cfg,
                snapshot_id=123,
                valid_features=features,
                asof_utc=ASOF_UTC,
                session_enum=SessionState.RTH,
                sec_to_close=None,
                endpoint_coverage=1.0,
            )
    assert len(predictions) == 1
    return predictions[0]


def test_ood_uses_structured_expected_bounds_not_units_expected_text():
    features = [
        _feature("spot", 150.0, path="/api/stock/{ticker}/ohlc/{candle_size}", units_expected="Spot Price"),
        _feature(
            "oi_pressure",
            1.2,
            path="/api/stock/{ticker}/oi-per-strike",
            bounded_output=True,
            expected_bounds=(-1.0, 1.0),
            emitted_units="directional_imbalance_ratio",
            raw_input_units="Open Interest (contracts)",
            output_domain="closed_interval",
            output_domain_contract_version="output_domain/v1",
            units_expected="Open Interest (contracts)",
        ),
    ]

    assessment = assess_operational_ood(
        feature_rows=features,
        decision_feature_keys=["spot", "oi_pressure"],
        session_state=SessionState.RTH,
    )

    assert assessment.state == OODState.OUT_OF_DISTRIBUTION
    assert assessment.primary_reason == "boundary_violation:oi_pressure"
    violation = assessment.boundary_violation_features["oi_pressure"]
    assert violation["boundary_source"] == "structured"
    assert violation["upper_bound"] == 1.0
    assert violation["emitted_units"] == "directional_imbalance_ratio"


def test_ood_degrades_when_bounded_decision_feature_lacks_expected_bounds():
    features = [
        _feature("spot", 150.0, path="/api/stock/{ticker}/ohlc/{candle_size}", units_expected="Spot Price"),
        _feature(
            "oi_pressure",
            0.4,
            path="/api/stock/{ticker}/oi-per-strike",
            bounded_output=True,
            expected_bounds=None,
            emitted_units="directional_imbalance_ratio",
            raw_input_units="Open Interest (contracts)",
            output_domain="closed_interval",
            output_domain_contract_version="output_domain/v1",
            units_expected="Directional Imbalance Ratio [-1, 1]",
        ),
    ]

    assessment = assess_operational_ood(
        feature_rows=features,
        decision_feature_keys=["spot", "oi_pressure"],
        session_state=SessionState.RTH,
    )

    assert assessment.state == OODState.DEGRADED
    assert assessment.primary_reason.startswith("output_domain_contract_missing:oi_pressure")
    assert assessment.output_domain_missing_features == ("oi_pressure",)
    assert assessment.output_domain_contract_issues["oi_pressure"]["missing_fields"] == ["expected_bounds"]


def test_ood_rejects_discrete_sign_value_outside_allowed_values():
    features = [
        _feature("spot", 150.0, path="/api/stock/{ticker}/ohlc/{candle_size}", units_expected="Spot Price"),
        _feature(
            "net_gex_sign",
            0.5,
            path="/api/stock/{ticker}/spot-exposures",
            bounded_output=True,
            expected_bounds=(-1.0, 1.0),
            emitted_units="directional_sign",
            raw_input_units="Gamma Exposure (provider aggregate units)",
            output_domain="discrete_sign",
            output_domain_contract_version="output_domain/v1",
            allowed_values=[-1.0, 0.0, 1.0],
            units_expected="Gamma Sign",
        ),
    ]

    assessment = assess_operational_ood(
        feature_rows=features,
        decision_feature_keys=["spot", "net_gex_sign"],
        session_state=SessionState.RTH,
    )

    assert assessment.state == OODState.OUT_OF_DISTRIBUTION
    assert assessment.primary_reason == "boundary_violation:net_gex_sign"
    violation = assessment.boundary_violation_features["net_gex_sign"]
    assert violation["violation_type"] == "allowed_values"
    assert violation["allowed_values"] == [-1.0, 0.0, 1.0]


def test_generate_predictions_logs_output_domain_contract_missing(caplog):
    pred = _run(
        [
            _feature("spot", 150.0, path="/api/stock/{ticker}/ohlc/{candle_size}", units_expected="Spot Price"),
            _feature(
                "oi_pressure",
                0.25,
                path="/api/stock/{ticker}/oi-per-strike",
                bounded_output=True,
                expected_bounds=None,
                emitted_units="directional_imbalance_ratio",
                raw_input_units="Open Interest (contracts)",
                output_domain="closed_interval",
                output_domain_contract_version="output_domain/v1",
                units_expected="Directional Imbalance Ratio [-1, 1]",
            ),
        ],
        caplog=caplog,
    )

    assert pred["meta_json"]["ood_state"] == OODState.DEGRADED.value
    assert pred["meta_json"]["ood_reason"].startswith("output_domain_contract_missing:oi_pressure")
    assert pred["meta_json"]["ood_assessment"]["output_domain_missing_features"] == ["oi_pressure"]
    counters = [getattr(record, "counter", None) for record in caplog.records]
    assert "ood_degraded_count" in counters
    assert "ood_output_domain_missing_count" in counters
