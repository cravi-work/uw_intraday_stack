import io
import json
import logging
from contextlib import contextmanager
from datetime import datetime, timezone

from src.endpoint_truth import EndpointContext
from src.features import _build_contract_normalization_failure_bundle
from src.freshness_policy import (
    EndpointCriticality,
    FeatureDecisionReason,
    FeaturePolicyAssessment,
    LagClass,
    PolicyAction,
    ResolvedFeaturePolicy,
)
from src.ingest_engine import _log_feature_policy_assessment
from src.instruments import ContractNormalizationSummary
from src.logging_config import JsonFormatter, LogContext, log_prediction_decision
from src.storage import DbWriter

UTC = timezone.utc


@contextmanager
def capture_logger(module_logger: logging.Logger):
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter(LogContext(service="test_service", env="test")))

    old_handlers = list(module_logger.handlers)
    old_level = module_logger.level
    old_propagate = module_logger.propagate
    module_logger.handlers = [handler]
    module_logger.setLevel(logging.INFO)
    module_logger.propagate = False
    try:
        yield stream
    finally:
        module_logger.handlers = old_handlers
        module_logger.setLevel(old_level)
        module_logger.propagate = old_propagate


def parse_log_lines(stream: io.StringIO):
    return [json.loads(line) for line in stream.getvalue().splitlines() if line.strip()]


def test_normalization_failure_log_is_structured():
    from src import features as feat_mod

    ctx = EndpointContext(
        endpoint_id=7,
        method="GET",
        path="/api/stock/SPY/oi-per-strike",
        operation_id=None,
        signature="sig",
        used_event_id=None,
        payload_class="SUCCESS_DATA",
        freshness_state="FRESH",
        stale_age_min=0,
        na_reason=None,
    )
    summary = ContractNormalizationSummary(
        status="INVALID",
        results=[],
        normalized_rows=[],
        required_row_count=1,
        invalid_row_count=1,
        adjusted_contract_count=0,
        duplicate_contract_keys=0,
        failure_reason="multiplier_mismatch",
    )

    with capture_logger(feat_mod.logger) as stream:
        bundle = _build_contract_normalization_failure_bundle(
            ctx,
            "extract_oi",
            {"metric_name": "oi_pressure"},
            payload={"rows": []},
            feature_keys=["oi_pressure"],
            meta_key="oi",
            summary=summary,
        )

    payload = parse_log_lines(stream)[0]
    assert bundle.features["oi_pressure"] is None
    assert payload["service"] == "test_service"
    assert payload["event"] == "normalization_failure"
    assert payload["counter"] == "normalization_failure_count"
    assert payload["feature_key"] == "oi"
    assert payload["reason"] == "multiplier_mismatch"
    assert payload["extractor"] == "extract_oi"


def test_feature_policy_assessment_logs_join_skew_and_stale_endpoint_events():
    from src import ingest_engine as ie_mod

    policy = ResolvedFeaturePolicy(
        name="live_ohlc_price",
        max_tolerated_age_seconds=60,
        join_skew_tolerance_seconds=60,
        criticality=EndpointCriticality.CRITICAL,
        lag_class=LagClass.LIVE,
        fresh_behavior=PolicyAction.ACCEPT,
        stale_behavior=PolicyAction.SUPPRESS,
        carry_forward_behavior=PolicyAction.SUPPRESS,
        empty_valid_behavior=PolicyAction.ACCEPT,
        time_provenance_degraded_behavior=PolicyAction.SUPPRESS,
        sources=("/api/stock/{ticker}/ohlc/",),
    )
    join_skew = FeaturePolicyAssessment(
        feature_key="spot",
        policy=policy,
        include_in_scoring=False,
        degraded=False,
        reason=FeatureDecisionReason.JOIN_SKEW_VIOLATION,
        reason_detail="delta_exceeds_tolerance",
        dq_reason_code="spot_join_skew_violation",
        effective_ts=datetime(2026, 3, 3, 14, 31, tzinfo=UTC),
        delta_seconds=121,
        normalized_future_ts=False,
        freshness_state="FRESH",
        stale_age_seconds=None,
        stale_age_minutes=None,
        time_provenance_degraded=False,
        policy_source="unit_test",
    )
    stale = FeaturePolicyAssessment(
        feature_key="spot",
        policy=policy,
        include_in_scoring=False,
        degraded=False,
        reason=FeatureDecisionReason.STALE_ENDPOINT_REJECTED,
        reason_detail="older_than_registry_max_age",
        dq_reason_code="spot_stale_endpoint_rejected",
        effective_ts=datetime(2026, 3, 3, 14, 25, tzinfo=UTC),
        delta_seconds=420,
        normalized_future_ts=False,
        freshness_state="STALE_CARRY",
        stale_age_seconds=420,
        stale_age_minutes=7,
        time_provenance_degraded=False,
        policy_source="unit_test",
    )

    with capture_logger(ie_mod.logger) as stream:
        _log_feature_policy_assessment(join_skew)
        _log_feature_policy_assessment(stale)

    events = parse_log_lines(stream)
    by_event = {event["event"]: event for event in events}
    assert by_event["join_skew_failure"]["counter"] == "join_skew_violation_count"
    assert by_event["join_skew_failure"]["delta_sec"] == 121
    assert by_event["join_skew_failure"]["policy"] == "live_ohlc_price"
    assert by_event["stale_endpoint"]["counter"] == "stale_endpoint_rejection_count"
    assert by_event["stale_endpoint"]["reason_detail"] == "older_than_registry_max_age"
    assert by_event["stale_endpoint"]["lag_class"] == "LIVE"


def test_prediction_decision_logs_include_contract_context_for_calibration_and_ood():
    logger = logging.getLogger("tests.observability.prediction")
    calibration_missing_prediction = {
        "snapshot_id": "snap-1",
        "horizon_kind": "FIXED",
        "horizon_minutes": 15,
        "decision_window_id": "window-1",
        "decision_state": "NO_SIGNAL",
        "risk_gate_status": "DEGRADED",
        "data_quality_state": "PARTIAL",
        "confidence_state": "LOW",
        "model_name": "bounded_additive_score",
        "model_version": "2.1.0",
        "meta_json": {
            "decision_dq": 0.37,
            "endpoint_coverage": 0.58,
            "prediction_contract": {
                "target_name": "intraday_return",
                "target_version": "v3",
                "label_contract": {
                    "label_version": "lbl_v3",
                    "threshold_policy_version": "thr_v2",
                },
                "target_spec": {
                    "target_name": "intraday_return",
                    "target_version": "v3",
                },
            },
            "probability_contract": {
                "suppression_reason": "MISSING_CALIBRATION_ARTIFACT",
                "ood_state": "UNKNOWN",
            },
        },
    }
    ood_prediction = {
        **calibration_missing_prediction,
        "snapshot_id": "snap-2",
        "meta_json": {
            **calibration_missing_prediction["meta_json"],
            "probability_contract": {
                "suppression_reason": "OOD_REJECTION",
                "ood_state": "OUT_OF_DISTRIBUTION",
            },
        },
    }

    with capture_logger(logger) as stream:
        log_prediction_decision(
            logger,
            calibration_missing_prediction,
            ticker="SPY",
            asof_ts_utc=datetime(2026, 3, 3, 14, 35, tzinfo=UTC),
        )
        log_prediction_decision(
            logger,
            ood_prediction,
            ticker="QQQ",
            asof_ts_utc=datetime(2026, 3, 3, 14, 40, tzinfo=UTC),
        )

    events = parse_log_lines(stream)
    names = [event["event"] for event in events]
    assert "calibration_missing" in names
    assert "ood_rejection" in names
    assert names.count("signal_suppressed") == 2

    calibration_event = next(event for event in events if event["event"] == "calibration_missing")
    assert calibration_event["counter"] == "calibration_missing_count"
    assert calibration_event["ticker"] == "SPY"
    assert calibration_event["target_name"] == "intraday_return"
    assert calibration_event["target_version"] == "v3"
    assert calibration_event["label_version"] == "lbl_v3"
    assert calibration_event["threshold_policy_version"] == "thr_v2"
    assert calibration_event["model_version"] == "2.1.0"

    ood_event = next(event for event in events if event["event"] == "ood_rejection")
    assert ood_event["counter"] == "ood_rejection_count"
    assert ood_event["ticker"] == "QQQ"
    assert ood_event["ood_state"] == "OUT_OF_DISTRIBUTION"
    assert ood_event["suppression_reason"] == "OOD_REJECTION"


def test_insert_prediction_persists_decision_trace_and_diagnostics(tmp_path):
    db_path = tmp_path / "obs_contract.duckdb"
    writer = DbWriter(str(db_path))

    with writer.writer() as con:
        writer.ensure_schema(con)
        snapshot_id = writer.insert_snapshot(
            con,
            run_id=None,
            asof_ts_utc=datetime(2026, 3, 3, 14, 35, tzinfo=UTC),
            ticker="SPY",
            session_label="RTH",
            is_trading_day=True,
            is_early_close=False,
            data_quality_score=0.9,
            market_close_utc=datetime(2026, 3, 3, 21, 0, tzinfo=UTC),
            post_end_utc=datetime(2026, 3, 3, 23, 0, tzinfo=UTC),
            seconds_to_close=23400,
        )
        prediction_id = writer.insert_prediction(
            con,
            {
                "snapshot_id": snapshot_id,
                "horizon_kind": "FIXED",
                "horizon_minutes": 15,
                "start_price": 500.12,
                "bias": 0.0,
                "confidence": 0.0,
                "prob_up": None,
                "prob_down": None,
                "prob_flat": None,
                "model_name": "bounded_additive_score",
                "model_version": "2.1.0",
                "decision_state": "NO_SIGNAL",
                "risk_gate_status": "DEGRADED",
                "data_quality_state": "PARTIAL",
                "confidence_state": "LOW",
                "blocked_reasons": [],
                "degraded_reasons": ["missing_calibration"],
                "validation_eligible": False,
                "meta_json": {
                    "feature_version": "feat_contract_v1",
                    "prediction_contract": {
                        "target_name": "intraday_return",
                        "target_version": "v3",
                        "label_contract": {
                            "label_version": "lbl_v3",
                            "threshold_policy_version": "thr_v2",
                        },
                        "target_spec": {
                            "target_name": "intraday_return",
                            "target_version": "v3",
                        },
                    },
                    "probability_contract": {
                        "suppression_reason": "MISSING_CALIBRATION_ARTIFACT",
                        "ood_state": "UNKNOWN",
                    },
                },
            },
        )

        event_type, suppression_reason, model_version, target_version, trace_json = con.execute(
            """
            SELECT event_type, suppression_reason, model_version, target_version, trace_json
            FROM decision_traces
            WHERE prediction_id = ?
            """,
            [prediction_id],
        ).fetchone()
        diagnostics = writer.get_pipeline_diagnostics(con)

    trace_payload = json.loads(trace_json) if isinstance(trace_json, str) else trace_json
    assert event_type == "calibration_missing"
    assert suppression_reason == "MISSING_CALIBRATION_ARTIFACT"
    assert model_version == "2.1.0"
    assert target_version == "v3"
    assert trace_payload["target_name"] == "intraday_return"
    assert trace_payload["label_version"] == "lbl_v3"
    assert diagnostics["decision_traces_by_event"]["calibration_missing"] == 1
