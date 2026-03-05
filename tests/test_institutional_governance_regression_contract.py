import copy
import datetime as dt

import pytest

from src.features import extract_smart_whale_pressure
from src.endpoint_truth import EndpointContext
from src.ingest_engine import _validate_config, generate_predictions
from src.models import ReplayMode, SessionState

ASOF_UTC = dt.datetime(2026, 3, 3, 15, 0, tzinfo=dt.timezone.utc)


def _artifact(*, version: str, replay_mode: str, provenance: bool = True) -> dict:
    artifact = {
        "artifact_name": "bounded_additive_score_calibration",
        "artifact_version": version,
        "target_name": "intraday_direction_3class",
        "target_version": "institutional.target.v1",
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
    if provenance:
        artifact["provenance"] = {
            "trained_from_utc": "2025-01-02T14:30:00+00:00",
            "trained_to_utc": "2025-12-31T21:00:00+00:00",
            "valid_from_utc": "2026-01-02T14:30:00+00:00",
            "valid_to_utc": "2026-12-31T21:00:00+00:00",
            "evidence_ref": f"replay://tests/{version}",
            "fit_sample_count": 20000,
        }
    return artifact



def _cfg(*, artifacts: list[dict], governance_mode: str = "FORWARD_OBSERVATION") -> dict:
    return {
        "ingestion": {"cadence_minutes": 5, "timezone": "America/New_York", "watchlist": ["AAPL"]},
        "storage": {"duckdb_path": "", "cycle_lock_path": "", "writer_lock_path": ""},
        "system": {"api_key_env": "UW_API_KEY"},
        "network": {"base_url": "https://api.unusualwhales.com"},
        "validation": {
            "horizon_weights_source": "explicit",
            "horizons_minutes": [15],
            "horizon_weights": {"15": {"spot": 1.0, "oi_pressure": 1.0}},
            "horizon_critical_features": {"15": ["spot"]},
            "use_default_required_features": False,
            "emit_to_close_horizon": False,
            "flat_threshold_pct": 0.001,
            "alignment_tolerance_sec": 900,
            "invalid_after_minutes": 60,
            "fallback_max_age_minutes": 15,
            "tolerance_minutes": 10,
            "max_horizon_drift_minutes": 10,
            "governance_mode": governance_mode,
            "decision_path_policy": {
                "contract_version": "decision_path/v1",
                "zero_weight_is_non_decision": True,
                "require_feature_metadata": True,
                "allow_explicit_zero_weight_critical_override": True,
                "explicit_zero_weight_critical_features": {"15": [], "60": [], "to_close": []},
            },
            "output_domain_policy": {
                "contract_version": "output_domain_policy/v1",
                "required_contract_version": "output_domain/v1",
                "require_bounded_output_contract": True,
                "require_expected_bounds": True,
                "require_emitted_units": True,
                "require_raw_input_units": True,
                "require_output_domain": True,
                "require_bounded_output_flag": True,
                "degrade_on_missing_contract": True,
                "enforce_on_decision_eligible_only": True,
            },
        },
        "model": {
            "model_name": "bounded_additive_score",
            "model_version": "institutional.model.v1",
            "target_spec": {
                "target_name": "intraday_direction_3class",
                "target_version": "institutional.target.v1",
            },
            "confidence_cap": 0.55,
            "min_confidence": 0.35,
            "neutral_threshold": 0.55,
            "direction_margin": 0.08,
            "min_flat_prob": 0.15,
            "max_flat_prob": 0.65,
            "flat_from_data_quality_scale": 0.9,
            "ood_assessment_policy": {
                "contract_version": "ood_assessment/v1",
                "degraded_coverage_threshold": 0.85,
                "out_coverage_threshold": 0.50,
                "boundary_slack": 1.0e-06,
                "require_assessment_before_emission": True,
            },
            "ood_probability_policy": {
                "contract_version": "ood_probability/v1",
                "out_confidence_scale": 0.0,
                "out_emit_calibrated": False,
                "unknown_confidence_scale": 0.0,
                "unknown_emit_calibrated": False,
                "degraded_confidence_scale": 0.50,
                "degraded_emit_calibrated": True,
            },
            "calibration_registry": {
                "contract_version": "calibration_registry/v1",
                "registry_version": "2026.03.0",
                "default_regime": "DEFAULT",
                "selection_policy": {
                    "require_scope_match": True,
                    "allow_legacy_fallback": False,
                    "allow_generic_scope_fallback": False,
                    "require_provenance": True,
                    "required_provenance_fields": [
                        "trained_from_utc",
                        "trained_to_utc",
                        "valid_from_utc",
                        "valid_to_utc",
                        "evidence_ref",
                        "fit_sample_count",
                    ],
                    "institutional_grade": governance_mode == "INSTITUTIONAL_GRADE",
                },
                "compatibility_rules": {
                    "require_target_match": True,
                    "require_horizon_match": True,
                    "require_session_match": True,
                    "require_regime_match": True,
                    "require_replay_mode_match": True,
                    "require_artifact_hash": True,
                    "require_provenance_fields": True,
                    "required_provenance_fields": [
                        "trained_from_utc",
                        "trained_to_utc",
                        "valid_from_utc",
                        "valid_to_utc",
                        "evidence_ref",
                        "fit_sample_count",
                    ],
                    "allow_generic_scope_fallback": False,
                },
                "artifacts": artifacts,
            },
        },
        "adapt": {"enabled": False},
    }



def _feature(
    feature_key: str,
    value: float | None,
    *,
    path: str,
    bounded_output: bool = False,
    expected_bounds: tuple[float, float] | None = None,
    emitted_units: str | None = None,
    raw_input_units: str | None = None,
    output_domain: str | None = None,
    units_expected: str,
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
                "expected_bounds": None
                if expected_bounds is None
                else {"lower": float(expected_bounds[0]), "upper": float(expected_bounds[1]), "inclusive": True},
                "emitted_units": emitted_units,
                "raw_input_units": raw_input_units,
                "output_domain": output_domain,
                "output_domain_contract_version": "output_domain/v1",
            }
        )
    else:
        lineage.update(
            {
                "bounded_output": False,
                "emitted_units": "Spot Price",
                "raw_input_units": "Spot Price",
                "output_domain": "unbounded_real",
                "output_domain_contract_version": "output_domain/v1",
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



def _run(cfg: dict, features: list[dict], *, replay_mode: ReplayMode) -> dict:
    predictions = generate_predictions(
        cfg,
        snapshot_id=9001,
        valid_features=features,
        asof_utc=ASOF_UTC,
        session_enum=SessionState.RTH,
        sec_to_close=None,
        endpoint_coverage=1.0,
        replay_mode=replay_mode,
    )
    assert len(predictions) == 1
    return predictions[0]



def test_replay_mode_specific_artifact_selection_and_mismatch_are_behavioral():
    cfg = _cfg(
        artifacts=[
            _artifact(version="cal.live", replay_mode="LIVE_LIKE_OBSERVED"),
            _artifact(version="cal.research", replay_mode="RESEARCH_RESTATED"),
        ]
    )
    features = [
        _feature("spot", 150.0, path="/api/stock/{ticker}/ohlc/{candle_size}", units_expected="Spot Price"),
        _feature(
            "oi_pressure",
            0.25,
            path="/api/stock/{ticker}/oi-per-strike",
            bounded_output=True,
            expected_bounds=(-1.0, 1.0),
            emitted_units="directional_imbalance_ratio",
            raw_input_units="Open Interest (contracts)",
            output_domain="closed_interval",
            units_expected="Directional Imbalance Ratio [-1, 1]",
        ),
    ]

    live_like = _run(cfg, features, replay_mode=ReplayMode.LIVE_LIKE_OBSERVED)
    research = _run(cfg, features, replay_mode=ReplayMode.RESEARCH_RESTATED)

    assert live_like["meta_json"]["calibration_selection"]["artifact"]["artifact_version"] == "cal.live"
    assert research["meta_json"]["calibration_selection"]["artifact"]["artifact_version"] == "cal.research"
    assert live_like["meta_json"]["prediction_contract"]["prediction_replay_mode"] == "LIVE_LIKE_OBSERVED"
    assert research["meta_json"]["prediction_contract"]["prediction_replay_mode"] == "RESEARCH_RESTATED"
    assert live_like["prob_up"] is not None
    assert research["prob_up"] is not None

    mismatch_cfg = _cfg(artifacts=[_artifact(version="cal.research", replay_mode="RESEARCH_RESTATED")])
    mismatch = _run(mismatch_cfg, features, replay_mode=ReplayMode.LIVE_LIKE_OBSERVED)
    assert mismatch["meta_json"]["calibration_selection"]["reason_code"] == "REPLAY_MODE_MISMATCH"
    assert mismatch["meta_json"]["prediction_contract"]["calibration_request_replay_mode"] == "LIVE_LIKE_OBSERVED"
    assert mismatch["prob_up"] is None
    assert mismatch["meta_json"]["suppression_reason"] == "MISSING_CALIBRATION_ARTIFACT"



def test_bounded_feature_with_prose_units_only_degrades_ood_instead_of_passing():
    cfg = _cfg(artifacts=[_artifact(version="cal.live", replay_mode="LIVE_LIKE_OBSERVED")])
    features = [
        _feature("spot", 150.0, path="/api/stock/{ticker}/ohlc/{candle_size}", units_expected="Spot Price"),
        _feature(
            "oi_pressure",
            0.40,
            path="/api/stock/{ticker}/oi-per-strike",
            bounded_output=True,
            expected_bounds=None,
            emitted_units="directional_imbalance_ratio",
            raw_input_units="Open Interest (contracts)",
            output_domain="closed_interval",
            units_expected="Directional Imbalance Ratio [-1, 1]",
        ),
    ]

    pred = _run(cfg, features, replay_mode=ReplayMode.LIVE_LIKE_OBSERVED)

    assert pred["meta_json"]["ood_state"] == "DEGRADED"
    assert pred["meta_json"]["ood_reason"].startswith("output_domain_contract_missing:oi_pressure")
    assert pred["meta_json"]["ood_assessment"]["output_domain_missing_features"] == ["oi_pressure"]
    assert pred["meta_json"]["ood_assessment"]["output_domain_contract_issues"]["oi_pressure"]["missing_fields"] == ["expected_bounds"]



def test_institutional_mode_rejects_artifact_without_replay_specific_scope_or_provenance():
    cfg = _cfg(
        artifacts=[_artifact(version="cal.any", replay_mode="ANY", provenance=False)],
        governance_mode="INSTITUTIONAL_GRADE",
    )

    with pytest.raises((KeyError, ValueError)) as excinfo:
        _validate_config(copy.deepcopy(cfg))

    assert (
        "scope.replay_mode may not be ANY" in str(excinfo.value)
        or "Missing model.calibration_registry.artifacts[0]" in str(excinfo.value)
    )



def test_bounded_extractor_lineage_publishes_machine_readable_output_domain_contract():
    ctx = EndpointContext(
        endpoint_id=1,
        method="GET",
        path="/api/stock/{ticker}/flow-recent",
        operation_id="test_op",
        signature="GET /api/stock/{ticker}/flow-recent",
        used_event_id="evt-1",
        payload_class="SUCCESS_HAS_DATA",
        freshness_state="FRESH",
        stale_age_min=0,
        na_reason=None,
        endpoint_asof_ts_utc=ASOF_UTC,
        effective_ts_utc=ASOF_UTC,
        as_of_time_utc=ASOF_UTC,
        processed_at_utc=ASOF_UTC,
        received_at_utc=ASOF_UTC,
        effective_time_source="endpoint_provenance",
        timestamp_quality="VALID",
        endpoint_name="flow_recent",
        endpoint_purpose="signal-critical",
        decision_path=True,
        missing_affects_confidence=True,
        stale_affects_confidence=True,
        purpose_contract_version="endpoint_purpose/v1",
    )
    payload = [
        {
            "underlying": "AAPL",
            "expiration": "2026-06-19",
            "strike": 150.0,
            "put_call": "CALL",
            "premium": 60_000.0,
            "dte": 5.0,
            "side": "BUY",
            "multiplier": 100.0,
            "deliverable_shares": 100.0,
        }
    ]

    bundle = extract_smart_whale_pressure(payload, ctx)
    lineage = bundle.meta["flow"]["metric_lineage"]

    assert bundle.features["smart_whale_pressure"] is not None
    assert lineage["bounded_output"] is True
    assert lineage["output_domain_contract_version"] == "output_domain/v1"
    assert lineage["emitted_units"] == "normalized_directional_pressure"
    assert lineage["raw_input_units"] == "Net Premium Flow (USD)"
    assert lineage["expected_bounds"] == {"lower": -1.0, "upper": 1.0, "inclusive": True}
    assert lineage["units_expected"] == "Normalized Directional Pressure [-1, 1]"
