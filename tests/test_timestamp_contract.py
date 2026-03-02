import datetime as dt

from src.endpoint_truth import (
    EndpointPayloadClass,
    EndpointStateRow,
    FreshnessState,
    NaReasonCode,
    PayloadAssessment,
    resolve_effective_payload,
)
from src.endpoint_rules import EmptyPayloadPolicy
from src.ingest_engine import generate_predictions
from src.models import SessionState


def test_fresh_payload_without_provider_time_degrades_instead_of_using_asof():
    fixed_asof_utc = dt.datetime(2026, 1, 1, 12, 0, tzinfo=dt.timezone.utc)

    assessment = PayloadAssessment(
        payload_class=EndpointPayloadClass.SUCCESS_HAS_DATA,
        changed=True,
        missing_keys=[],
        error_reason=None,
        empty_policy=EmptyPayloadPolicy.EMPTY_IS_DATA,
        is_empty=False,
        validator=None,
    )

    resolved = resolve_effective_payload(
        current_event_id="mock_event_1",
        current_ts_raw=fixed_asof_utc,
        assessment=assessment,
        prev_state=None,
        fallback_max_age_seconds=900,
        as_of_time_raw=fixed_asof_utc,
    )

    assert resolved.freshness_state == FreshnessState.FRESH
    assert resolved.effective_ts_utc is None
    assert resolved.as_of_time_utc == fixed_asof_utc
    assert resolved.timestamp_quality == "DEGRADED"
    assert resolved.effective_time_source == "missing_provider_time"
    assert NaReasonCode.MISSING_PROVIDER_TIME.value in str(resolved.na_reason)
    assert NaReasonCode.TIME_PROVENANCE_DEGRADED.value in str(resolved.na_reason)


def test_documented_contemporaneous_endpoint_can_opt_in_to_asof_effective_time():
    fixed_asof_utc = dt.datetime(2026, 1, 1, 12, 0, tzinfo=dt.timezone.utc)

    assessment = PayloadAssessment(
        payload_class=EndpointPayloadClass.SUCCESS_HAS_DATA,
        changed=True,
        missing_keys=[],
        error_reason=None,
        empty_policy=EmptyPayloadPolicy.EMPTY_IS_DATA,
        is_empty=False,
        validator=None,
    )

    resolved = resolve_effective_payload(
        current_event_id="mock_event_1",
        current_ts_raw=fixed_asof_utc,
        assessment=assessment,
        prev_state=None,
        fallback_max_age_seconds=900,
        as_of_time_raw=fixed_asof_utc,
        documented_asof_contemporaneous=True,
    )

    assert resolved.effective_ts_utc == fixed_asof_utc
    assert resolved.timestamp_quality == "DEGRADED"
    assert resolved.effective_time_source == "documented_asof_contemporaneous"


def test_stale_carry_effective_ts_equals_last_success_ts():
    fixed_asof_utc = dt.datetime(2026, 1, 1, 12, 0, tzinfo=dt.timezone.utc)
    historical_success_utc = dt.datetime(2026, 1, 1, 11, 55, tzinfo=dt.timezone.utc)

    assessment = PayloadAssessment(
        payload_class=EndpointPayloadClass.ERROR,
        changed=False,
        missing_keys=[],
        error_reason="HTTP_500",
        empty_policy=EmptyPayloadPolicy.EMPTY_IS_DATA,
        is_empty=True,
        validator=None,
    )

    prev_state = EndpointStateRow(
        last_success_event_id="mock_event_0",
        last_success_ts_utc=historical_success_utc,
        last_payload_hash="hash",
        last_change_ts_utc=historical_success_utc,
        last_change_event_id="mock_event_0",
    )

    resolved = resolve_effective_payload(
        current_event_id="mock_event_1",
        current_ts_raw=fixed_asof_utc,
        assessment=assessment,
        prev_state=prev_state,
        fallback_max_age_seconds=900,
        as_of_time_raw=fixed_asof_utc,
    )

    assert resolved.freshness_state == FreshnessState.STALE_CARRY
    assert resolved.effective_ts_utc == historical_success_utc
    assert resolved.effective_ts_utc < fixed_asof_utc
    assert resolved.timestamp_quality == "LAGGED"
    assert resolved.lagged is True


def test_future_effective_ts_blocks_predictive_use():
    fixed_asof_utc = dt.datetime(2026, 1, 1, 12, 0, tzinfo=dt.timezone.utc)
    future_utc = fixed_asof_utc + dt.timedelta(minutes=10)

    cfg = {
        "ingestion": {"cadence_minutes": 5},
        "validation": {
            "invalid_after_minutes": 60,
            "tolerance_minutes": 10,
            "max_horizon_drift_minutes": 10,
            "flat_threshold_pct": 0.001,
            "fallback_max_age_minutes": 15,
            "alignment_tolerance_sec": 900,
            "use_default_required_features": False,
            "emit_to_close_horizon": False,
            "horizons_minutes": [15],
            "horizon_weights_source": "explicit",
            "horizon_critical_features": {"15": ["spot"]},
            "horizon_weights": {"15": {"spot": 1.0}},
        },
    }

    valid_meta = {"source_endpoints": [], "freshness_state": "FRESH", "stale_age_min": 0, "na_reason": None, "details": {}}
    features = [
        {"feature_key": "spot", "feature_value": 150.0, "meta_json": {**valid_meta, "metric_lineage": {"effective_ts_utc": future_utc.isoformat()}}}
    ]

    preds = generate_predictions(cfg, 1, features, fixed_asof_utc, SessionState.RTH, 3600, 1.0)
    p15 = next(p for p in preds if p["horizon_minutes"] == 15)

    assert p15["decision_state"] == "NO_SIGNAL"
    assert p15["alignment_status"] == "MISALIGNED"
    assert "spot" in p15["meta_json"]["alignment_diagnostics"]["future_ts_keys"]

    reasons_str = str(p15["blocked_reasons"])
    assert "no_horizon_target_features_after_alignment" in reasons_str or "spot_future_ts" in reasons_str


def test_one_sided_alignment_tolerance():
    fixed_asof_utc = dt.datetime(2026, 1, 1, 12, 0, tzinfo=dt.timezone.utc)
    tolerance_sec = 300

    past_ts_pass = fixed_asof_utc - dt.timedelta(seconds=tolerance_sec)
    past_ts_fail = fixed_asof_utc - dt.timedelta(seconds=tolerance_sec + 1)

    cfg = {
        "ingestion": {"cadence_minutes": 5},
        "validation": {
            "invalid_after_minutes": 60,
            "tolerance_minutes": 10,
            "max_horizon_drift_minutes": 10,
            "flat_threshold_pct": 0.001,
            "fallback_max_age_minutes": 15,
            "alignment_tolerance_sec": tolerance_sec,
            "use_default_required_features": False,
            "emit_to_close_horizon": False,
            "horizons_minutes": [15],
            "horizon_weights_source": "explicit",
            "horizon_critical_features": {"15": ["spot"]},
            "horizon_weights": {"15": {"spot": 1.0}},
        },
    }

    valid_meta = {"source_endpoints": [], "freshness_state": "FRESH", "stale_age_min": 0, "na_reason": None, "details": {}}

    features_pass = [
        {"feature_key": "spot", "feature_value": 150.0, "meta_json": {**valid_meta, "metric_lineage": {"effective_ts_utc": past_ts_pass.isoformat()}}}
    ]
    preds_pass = generate_predictions(cfg, 1, features_pass, fixed_asof_utc, SessionState.RTH, 3600, 1.0)
    p15_pass = next(p for p in preds_pass if p["horizon_minutes"] == 15)

    assert p15_pass["decision_state"] != "NO_SIGNAL"
    assert p15_pass["alignment_status"] == "ALIGNED"
    assert p15_pass["meta_json"]["alignment_diagnostics"]["excluded_misaligned_count"] == 0

    features_fail = [
        {"feature_key": "spot", "feature_value": 150.0, "meta_json": {**valid_meta, "metric_lineage": {"effective_ts_utc": past_ts_fail.isoformat()}}}
    ]
    preds_fail = generate_predictions(cfg, 1, features_fail, fixed_asof_utc, SessionState.RTH, 3600, 1.0)
    p15_fail = next(p for p in preds_fail if p["horizon_minutes"] == 15)

    assert p15_fail["decision_state"] == "NO_SIGNAL"
    assert p15_fail["alignment_status"] == "MISALIGNED"
    assert p15_fail["meta_json"]["alignment_diagnostics"]["excluded_misaligned_count"] == 1
    assert any("spot_delta_301s" in key for key in p15_fail["meta_json"]["alignment_diagnostics"]["misaligned_keys"])
