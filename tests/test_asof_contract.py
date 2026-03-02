import datetime as dt

from src.endpoint_rules import EmptyPayloadPolicy
from src.endpoint_truth import EndpointPayloadClass, PayloadAssessment, resolve_effective_payload


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
