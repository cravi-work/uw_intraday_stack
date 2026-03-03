import datetime as dt

from src.endpoint_rules import EmptyPayloadPolicy
from src.endpoint_truth import EndpointPayloadClass, EndpointStateRow, FreshnessState, PayloadAssessment, resolve_effective_payload


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
