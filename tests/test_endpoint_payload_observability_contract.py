import json
import logging
from contextlib import contextmanager
from io import StringIO
from logging import StreamHandler

from src.endpoint_truth import EndpointContext
from src.logging_config import JsonFormatter, LogContext
from src import features as features_mod


@contextmanager
def capture_logger(logger: logging.Logger):
    """Capture structured_log JSON output for assertions."""
    stream = StringIO()
    handler = StreamHandler(stream)
    handler.setFormatter(JsonFormatter(ctx=LogContext(service="uw_intraday_stack", env="test")))
    old_handlers = list(logger.handlers)
    old_level = logger.level

    logger.handlers = [handler]
    logger.setLevel(logging.INFO)
    try:
        yield stream
    finally:
        logger.handlers = old_handlers
        logger.setLevel(old_level)


def _read_events(stream: StringIO):
    stream.seek(0)
    events = []
    for line in stream.read().splitlines():
        if not line.strip():
            continue
        events.append(json.loads(line))
    return events


def test_endpoint_payload_observability_emits_required_fields_for_unusable_payload():
    """Contract: 200 OK can still be unusable (empty, time-missing, etc.).

    We require a structured, per-endpoint-family observability event with:
      * parsed_row_count
      * provider_timestamp_found_count
      * effective_timestamp_source
      * rows_discarded_before_extraction
      * feature_output_reason
    """

    ctx = EndpointContext(
        endpoint_id=1,
        endpoint_name="flow_recent",
        method="GET",
        path="/api/stock/{ticker}/flow-recent",
        operation_id="flow_recent",
        signature="GET /api/stock/{ticker}/flow-recent",
        used_event_id="ev1",
        payload_class="SUCCESS_EMPTY_VALID",
        freshness_state="EMPTY_VALID",
        stale_age_min=None,
        na_reason=None,
        effective_time_source="missing_provider_time",
        time_provenance_degraded=True,
        decision_path=True,
    )

    with capture_logger(features_mod.logger) as stream:
        features_mod.extract_all(effective_payloads={1: []}, contexts={1: ctx})

    events = _read_events(stream)
    obs = [e for e in events if e.get("event") == "endpoint_payload_observability"]
    assert len(obs) == 1, f"Expected 1 observability event, got {len(obs)}: {events}"

    e = obs[0]
    assert e.get("endpoint_name") == "flow_recent"
    assert e.get("endpoint_family") == "FLOW"

    # Required contract fields
    assert "parsed_row_count" in e
    assert "provider_timestamp_found_count" in e
    assert "effective_timestamp_source" in e
    assert "rows_discarded_before_extraction" in e
    assert "feature_output_reason" in e

    # A few sanity checks
    assert e.get("payload_row_count") == 0
    assert e.get("provider_timestamp_found_count") == 0
    assert e.get("effective_timestamp_source") == "missing_provider_time"
