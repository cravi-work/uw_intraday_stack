import datetime as dt
from unittest.mock import MagicMock

import pytest

from src.endpoint_truth import EndpointContext
from src.features import extract_price_features, extract_smart_whale_pressure


@pytest.fixture
def mock_ctx():
    ctx = MagicMock(spec=EndpointContext)
    ctx.freshness_state = "FRESH"
    ctx.na_reason = None
    ctx.path = "/api/mock"
    ctx.method = "GET"
    ctx.operation_id = "test"
    ctx.used_event_id = "mock-id"
    ctx.signature = "sig"
    ctx.endpoint_id = 1
    ctx.stale_age_min = 0
    ctx.effective_ts_utc = None
    ctx.event_time_utc = None
    ctx.source_publish_time_utc = None
    ctx.received_at_utc = dt.datetime(2026, 1, 1, 10, 0, 1, tzinfo=dt.timezone.utc)
    ctx.processed_at_utc = dt.datetime(2026, 1, 1, 10, 0, 2, tzinfo=dt.timezone.utc)
    ctx.as_of_time_utc = dt.datetime(2026, 1, 1, 10, 0, 0, tzinfo=dt.timezone.utc)
    ctx.source_revision = None
    ctx.effective_time_source = "missing_provider_time"
    ctx.timestamp_quality = "DEGRADED"
    ctx.lagged = False
    ctx.time_provenance_degraded = True
    return ctx


def test_payload_valid_timestamp_priority(mock_ctx):
    payload = [{"close": 150.0, "t": 1700000000.0}]
    bundle = extract_price_features(payload, mock_ctx)

    lineage = bundle.meta["price"]["metric_lineage"]
    assert lineage["timestamp_source"] == "payload_effective_time"
    assert lineage["timestamp_quality"] == "VALID"
    assert lineage["effective_ts_utc"] is not None
    assert lineage["event_time"] is not None
    assert lineage["processed_at"] == mock_ctx.processed_at_utc.isoformat()
    assert lineage["as_of_time"] == mock_ctx.as_of_time_utc.isoformat()


def test_payload_epoch_millis_timestamp_is_supported(mock_ctx):
    """Some UW endpoints emit epoch timestamps in milliseconds; those must not lose effective_ts_utc."""
    ts_s = 1700000000.0
    payload = [{"close": 150.0, "t": ts_s * 1000.0}]

    bundle = extract_price_features(payload, mock_ctx)
    lineage = bundle.meta["price"]["metric_lineage"]

    expected_iso = dt.datetime.fromtimestamp(ts_s, tz=dt.timezone.utc).isoformat()
    assert lineage["timestamp_source"] == "payload_effective_time"
    assert lineage["timestamp_quality"] == "VALID"
    assert lineage["effective_ts_utc"] == expected_iso


def test_payload_candle_start_end_time_timestamp_is_supported(mock_ctx):
    """UW OHLC Candle objects commonly provide `start_time`/`end_time` ISO8601 timestamps."""
    payload = [
        {
            "close": 150.0,
            "start_time": "2026-03-05T14:31:00Z",
            "end_time": "2026-03-05T14:32:00Z",
        }
    ]

    bundle = extract_price_features(payload, mock_ctx)
    lineage = bundle.meta["price"]["metric_lineage"]

    assert lineage["timestamp_source"] == "payload_effective_time"
    assert lineage["timestamp_quality"] == "VALID"
    assert lineage["effective_ts_utc"] == "2026-03-05T14:32:00+00:00"
    assert lineage["event_time"] == "2026-03-05T14:32:00+00:00"


def test_missing_provider_timestamp_is_explicitly_degraded(mock_ctx):
    payload = [{"premium": 15000.0, "dte": 5.0, "side": "ASK", "put_call": "CALL"}]
    bundle = extract_smart_whale_pressure(payload, mock_ctx)

    lineage = bundle.meta["flow"]["metric_lineage"]
    assert lineage["timestamp_source"] == "missing_provider_time"
    assert lineage["timestamp_quality"] == "DEGRADED"
    assert lineage["effective_ts_utc"] is None
    assert lineage["as_of_time"] == mock_ctx.as_of_time_utc.isoformat()
    assert lineage["time_provenance_degraded"] is True


def test_malformed_payload_timestamp_keeps_feature(mock_ctx):
    payload = [{"close": 150.0, "t": "invalid_date_string"}]
    bundle = extract_price_features(payload, mock_ctx)

    assert bundle.features["spot"] == 150.0

    lineage = bundle.meta["price"]["metric_lineage"]
    assert lineage["timestamp_source"] == "payload_effective_time"
    assert lineage["timestamp_quality"] == "INVALID"
    assert lineage["effective_ts_utc"] is None
