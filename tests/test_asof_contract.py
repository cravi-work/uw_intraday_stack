from unittest.mock import Mock
import datetime as dt

from src.endpoint_rules import EmptyPayloadPolicy
from src.endpoint_truth import EndpointPayloadClass, EndpointStateRow, FreshnessState, PayloadAssessment, infer_source_time_hints, resolve_effective_payload


def test_resolved_lineage_carries_explicit_time_fields():
    asof_utc = dt.datetime(2026, 1, 1, 12, 0, tzinfo=dt.timezone.utc)
    received_at = asof_utc - dt.timedelta(seconds=3)
    processed_at = asof_utc - dt.timedelta(seconds=1)
    event_time = asof_utc - dt.timedelta(seconds=30)

    assessment = PayloadAssessment(
        payload_class=EndpointPayloadClass.SUCCESS_HAS_DATA,
        empty_policy=EmptyPayloadPolicy.EMPTY_IS_DATA,
        is_empty=False,
        changed=True,
        error_reason=None,
    )

    resolved = resolve_effective_payload(
        current_event_id="evt-1",
        current_ts_raw=asof_utc,
        assessment=assessment,
        prev_state=None,
        as_of_time_raw=asof_utc,
        received_at_raw=received_at,
        processed_at_raw=processed_at,
        source_event_time_raw=event_time,
        source_revision="rev-7",
    )

    assert resolved.effective_ts_utc == event_time
    assert resolved.event_time_utc == event_time
    assert resolved.received_at_utc == received_at
    assert resolved.processed_at_utc == processed_at
    assert resolved.as_of_time_utc == asof_utc
    assert resolved.source_revision == "rev-7"
    assert resolved.effective_time_source == "event_time"
    assert resolved.timestamp_quality == "VALID"


def test_missing_provider_timestamp_never_silently_aligns_to_snapshot_time():
    asof_utc = dt.datetime(2026, 1, 1, 12, 0, tzinfo=dt.timezone.utc)

    assessment = PayloadAssessment(
        payload_class=EndpointPayloadClass.SUCCESS_HAS_DATA,
        empty_policy=EmptyPayloadPolicy.EMPTY_IS_DATA,
        is_empty=False,
        changed=True,
        error_reason=None,
    )

    resolved = resolve_effective_payload(
        current_event_id="evt-1",
        current_ts_raw=asof_utc,
        assessment=assessment,
        prev_state=None,
        as_of_time_raw=asof_utc,
        received_at_raw=asof_utc,
        processed_at_raw=asof_utc,
    )

    assert resolved.effective_ts_utc is None
    assert resolved.as_of_time_utc == asof_utc
    assert resolved.effective_time_source == "missing_provider_time"
    assert resolved.timestamp_quality == "DEGRADED"
    assert resolved.time_provenance_degraded is True



def test_documented_asof_contemporaneous_fallback_is_explicitly_degraded():
    asof_utc = dt.datetime(2026, 1, 1, 12, 0, tzinfo=dt.timezone.utc)

    assessment = PayloadAssessment(
        payload_class=EndpointPayloadClass.SUCCESS_HAS_DATA,
        empty_policy=EmptyPayloadPolicy.EMPTY_IS_DATA,
        is_empty=False,
        changed=True,
        error_reason=None,
    )

    resolved = resolve_effective_payload(
        current_event_id="evt-2",
        current_ts_raw=asof_utc,
        assessment=assessment,
        prev_state=None,
        as_of_time_raw=asof_utc,
        documented_asof_contemporaneous=True,
    )

    assert resolved.effective_ts_utc == asof_utc
    assert resolved.effective_time_source == "documented_asof_contemporaneous"
    assert resolved.timestamp_quality == "DEGRADED"
    assert resolved.time_provenance_degraded is True
    assert "MISSING_PROVIDER_TIME" in str(resolved.na_reason)
    assert "TIME_PROVENANCE_DEGRADED" in str(resolved.na_reason)



def test_source_publish_time_is_used_when_event_time_is_absent():
    asof_utc = dt.datetime(2026, 1, 1, 12, 0, tzinfo=dt.timezone.utc)
    publish_time = asof_utc - dt.timedelta(seconds=20)

    assessment = PayloadAssessment(
        payload_class=EndpointPayloadClass.SUCCESS_HAS_DATA,
        empty_policy=EmptyPayloadPolicy.EMPTY_IS_DATA,
        is_empty=False,
        changed=True,
        error_reason=None,
    )

    resolved = resolve_effective_payload(
        current_event_id="evt-3",
        current_ts_raw=asof_utc,
        assessment=assessment,
        prev_state=None,
        as_of_time_raw=asof_utc,
        source_publish_time_raw=publish_time,
    )

    assert resolved.effective_ts_utc == publish_time
    assert resolved.source_publish_time_utc == publish_time
    assert resolved.event_time_utc is None
    assert resolved.effective_time_source == "source_publish_time"
    assert resolved.timestamp_quality == "VALID"



def test_empty_means_stale_carry_forward_is_marked_lagged_not_live():
    asof_utc = dt.datetime(2026, 1, 1, 12, 0, tzinfo=dt.timezone.utc)
    prev_state = EndpointStateRow(
        last_success_event_id="evt-last-success",
        last_success_ts_utc=asof_utc - dt.timedelta(minutes=3),
        last_payload_hash="hash-1",
        last_change_ts_utc=asof_utc - dt.timedelta(minutes=5),
        last_change_event_id="evt-last-change",
    )
    assessment = PayloadAssessment(
        payload_class=EndpointPayloadClass.SUCCESS_EMPTY_VALID,
        empty_policy=EmptyPayloadPolicy.EMPTY_MEANS_STALE,
        is_empty=True,
        changed=False,
        error_reason="EMPTY_MEANS_STALE",
    )

    resolved = resolve_effective_payload(
        current_event_id="evt-4",
        current_ts_raw=asof_utc,
        assessment=assessment,
        prev_state=prev_state,
        as_of_time_raw=asof_utc,
    )

    assert resolved.freshness_state == FreshnessState.STALE_CARRY
    assert resolved.used_event_id == "evt-last-change"
    assert resolved.effective_time_source == "carry_forward_last_success"
    assert resolved.timestamp_quality == "LAGGED"
    assert resolved.lagged is True
    assert resolved.effective_ts_utc == prev_state.last_success_ts_utc


def test_nested_payload_provider_timestamps_reduce_degraded_fallback():
    asof_utc = dt.datetime(2026, 1, 1, 12, 0, tzinfo=dt.timezone.utc)
    payload = {
        "data": {
            "items": [
                {
                    "price": 150.0,
                    "executed_at": "2026-01-01T11:59:20+00:00",
                    "published_at": "2026-01-01T11:59:40+00:00",
                    "revision": "rev-nested-1",
                }
            ]
        }
    }

    hints = infer_source_time_hints(payload_json=payload)
    assessment = PayloadAssessment(
        payload_class=EndpointPayloadClass.SUCCESS_HAS_DATA,
        empty_policy=EmptyPayloadPolicy.EMPTY_IS_DATA,
        is_empty=False,
        changed=True,
        error_reason=None,
    )

    resolved = resolve_effective_payload(
        current_event_id="evt-nested",
        current_ts_raw=asof_utc,
        assessment=assessment,
        prev_state=None,
        as_of_time_raw=asof_utc,
        source_event_time_raw=hints.event_time_utc,
        source_publish_time_raw=hints.source_publish_time_utc,
        effective_time_raw=hints.effective_time_utc,
        source_revision=hints.source_revision,
    )

    assert hints.event_time_utc == dt.datetime(2026, 1, 1, 11, 59, 20, tzinfo=dt.timezone.utc)
    assert hints.source_publish_time_utc == dt.datetime(2026, 1, 1, 11, 59, 40, tzinfo=dt.timezone.utc)
    assert hints.source_revision == "rev-nested-1"
    assert resolved.effective_ts_utc == hints.event_time_utc
    assert resolved.effective_time_source == "event_time"
    assert resolved.timestamp_quality == "VALID"
    assert resolved.time_provenance_degraded is False


def test_response_header_publish_time_and_revision_are_used_when_payload_is_timestamp_poor():
    asof_utc = dt.datetime(2026, 1, 1, 12, 0, tzinfo=dt.timezone.utc)
    headers = {
        "Last-Modified": "Thu, 01 Jan 2026 11:59:30 GMT",
        "ETag": "rev-header-2",
    }

    hints = infer_source_time_hints(payload_json={"data": [{"close": 150.0}]}, response_headers=headers)
    assessment = PayloadAssessment(
        payload_class=EndpointPayloadClass.SUCCESS_HAS_DATA,
        empty_policy=EmptyPayloadPolicy.EMPTY_IS_DATA,
        is_empty=False,
        changed=True,
        error_reason=None,
    )

    resolved = resolve_effective_payload(
        current_event_id="evt-header",
        current_ts_raw=asof_utc,
        assessment=assessment,
        prev_state=None,
        as_of_time_raw=asof_utc,
        source_event_time_raw=hints.event_time_utc,
        source_publish_time_raw=hints.source_publish_time_utc,
        effective_time_raw=hints.effective_time_utc,
        source_revision=hints.source_revision,
    )

    assert hints.event_time_utc is None
    assert hints.source_publish_time_utc == dt.datetime(2026, 1, 1, 11, 59, 30, tzinfo=dt.timezone.utc)
    assert hints.source_revision == "rev-header-2"
    assert resolved.effective_ts_utc == hints.source_publish_time_utc
    assert resolved.source_publish_time_utc == hints.source_publish_time_utc
    assert resolved.source_revision == "rev-header-2"
    assert resolved.effective_time_source == "source_publish_time"
    assert resolved.timestamp_quality == "VALID"
    assert resolved.time_provenance_degraded is False


def test_explicit_provider_fields_override_inferred_payload_and_header_values():
    payload = {
        "data": [{
            "executed_at": "2026-01-01T11:58:00+00:00",
            "published_at": "2026-01-01T11:58:30+00:00",
            "revision": "rev-payload",
        }]
    }
    headers = {
        "Last-Modified": "Thu, 01 Jan 2026 11:58:45 GMT",
        "ETag": "rev-header",
    }

    hints = infer_source_time_hints(
        payload_json=payload,
        response_headers=headers,
        explicit_event_time_raw="2026-01-01T11:59:10+00:00",
        explicit_publish_time_raw="2026-01-01T11:59:20+00:00",
        explicit_effective_time_raw="2026-01-01T11:59:25+00:00",
        explicit_revision="rev-explicit",
    )

    assert hints.event_time_utc == dt.datetime(2026, 1, 1, 11, 59, 10, tzinfo=dt.timezone.utc)
    assert hints.source_publish_time_utc == dt.datetime(2026, 1, 1, 11, 59, 20, tzinfo=dt.timezone.utc)
    assert hints.effective_time_utc == dt.datetime(2026, 1, 1, 11, 59, 25, tzinfo=dt.timezone.utc)
    assert hints.source_revision == "rev-explicit"


def test_camel_case_payload_provider_time_fields_are_inferred_before_degraded_fallback():
    asof_utc = dt.datetime(2026, 1, 1, 12, 0, tzinfo=dt.timezone.utc)
    payload = {
        "meta": {
            "effectiveTime": "2026-01-01T11:59:50+00:00",
            "publishedAt": "2026-01-01T11:59:45+00:00",
            "sourceRevision": "rev-camel-1",
        },
        "data": [{"close": 150.0, "eventTime": "2026-01-01T11:59:40+00:00"}],
    }

    hints = infer_source_time_hints(payload_json=payload)
    assessment = PayloadAssessment(
        payload_class=EndpointPayloadClass.SUCCESS_HAS_DATA,
        empty_policy=EmptyPayloadPolicy.EMPTY_IS_DATA,
        is_empty=False,
        changed=True,
        error_reason=None,
    )

    resolved = resolve_effective_payload(
        current_event_id="evt-camel",
        current_ts_raw=asof_utc,
        assessment=assessment,
        prev_state=None,
        as_of_time_raw=asof_utc,
        source_event_time_raw=hints.event_time_utc,
        source_publish_time_raw=hints.source_publish_time_utc,
        effective_time_raw=hints.effective_time_utc,
        source_revision=hints.source_revision,
    )

    assert hints.event_time_utc == dt.datetime(2026, 1, 1, 11, 59, 40, tzinfo=dt.timezone.utc)
    assert hints.source_publish_time_utc == dt.datetime(2026, 1, 1, 11, 59, 45, tzinfo=dt.timezone.utc)
    assert hints.effective_time_utc == dt.datetime(2026, 1, 1, 11, 59, 50, tzinfo=dt.timezone.utc)
    assert hints.source_revision == "rev-camel-1"
    assert resolved.effective_ts_utc == hints.effective_time_utc
    assert resolved.effective_time_source == "payload_effective_time"
    assert resolved.timestamp_quality == "VALID"
    assert resolved.time_provenance_degraded is False



def test_response_header_event_and_effective_times_reduce_degraded_fallback():
    asof_utc = dt.datetime(2026, 1, 1, 12, 0, tzinfo=dt.timezone.utc)
    headers = {
        "X-Source-Event-Time": "2026-01-01T11:59:20+00:00",
        "X-Effective-Time": "2026-01-01T11:59:30+00:00",
        "X-Source-Revision": "rev-effective-header",
    }

    hints = infer_source_time_hints(payload_json={"data": [{"close": 151.0}]}, response_headers=headers)
    assessment = PayloadAssessment(
        payload_class=EndpointPayloadClass.SUCCESS_HAS_DATA,
        empty_policy=EmptyPayloadPolicy.EMPTY_IS_DATA,
        is_empty=False,
        changed=True,
        error_reason=None,
    )

    resolved = resolve_effective_payload(
        current_event_id="evt-effective-header",
        current_ts_raw=asof_utc,
        assessment=assessment,
        prev_state=None,
        as_of_time_raw=asof_utc,
        source_event_time_raw=hints.event_time_utc,
        source_publish_time_raw=hints.source_publish_time_utc,
        effective_time_raw=hints.effective_time_utc,
        source_revision=hints.source_revision,
    )

    assert hints.event_time_utc == dt.datetime(2026, 1, 1, 11, 59, 20, tzinfo=dt.timezone.utc)
    assert hints.effective_time_utc == dt.datetime(2026, 1, 1, 11, 59, 30, tzinfo=dt.timezone.utc)
    assert hints.source_revision == "rev-effective-header"
    assert resolved.effective_ts_utc == hints.effective_time_utc
    assert resolved.effective_time_source == "payload_effective_time"
    assert resolved.timestamp_quality == "VALID"
    assert resolved.time_provenance_degraded is False



def test_response_header_generated_at_is_used_when_payload_is_timestamp_poor():
    headers = {
        "X-Generated-At": "2026-01-01T11:59:30+00:00",
        "X-Data-Revision": "rev-generated-header",
    }

    hints = infer_source_time_hints(payload_json={"meta": {}, "data": [{"close": 151.0}]}, response_headers=headers)

    assert hints.event_time_utc is None
    assert hints.source_publish_time_utc == dt.datetime(2026, 1, 1, 11, 59, 30, tzinfo=dt.timezone.utc)
    assert hints.source_revision == "rev-generated-header"



def test_uw_client_preserves_provider_time_headers_for_downstream_inference():
    from src.uw_client import UwClient

    client = UwClient(
        registry=Mock(),
        base_url="https://example.com",
        api_key_env="UW_API_KEY",
        timeout_seconds=5,
        max_retries=0,
        backoff_seconds=1,
        max_concurrent_requests=1,
        rate_limit_per_second=1,
        circuit_failure_threshold=1,
        circuit_cool_down_seconds=1,
        circuit_half_open_max_calls=1,
    )

    headers = client._extract_response_headers({
        "X-Source-Event-Time": "2026-01-01T11:59:20+00:00",
        "X-Effective-Time": "2026-01-01T11:59:30+00:00",
        "X-Generated-At": "2026-01-01T11:59:40+00:00",
        "X-Data-Revision": "rev-provider-meta",
        "Ignored": "value",
    })

    assert headers["x-source-event-time"] == "2026-01-01T11:59:20+00:00"
    assert headers["x-effective-time"] == "2026-01-01T11:59:30+00:00"
    assert headers["x-generated-at"] == "2026-01-01T11:59:40+00:00"
    assert headers["x-data-revision"] == "rev-provider-meta"
    assert "ignored" not in headers
