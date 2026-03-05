import datetime as dt
from unittest.mock import MagicMock

import pytest

from src.endpoint_truth import EndpointContext
from src.features import extract_dealer_greeks, extract_gex_sign, extract_price_features, extract_smart_whale_pressure, extract_vol_skew


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


def test_price_feature_uses_row_date_when_t_is_missing(mock_ctx):
    payload = [{"close": 151.25, "date": "2026-01-01T10:00:30+00:00"}]
    bundle = extract_price_features(payload, mock_ctx)

    lineage = bundle.meta["price"]["metric_lineage"]
    assert bundle.features["spot"] == 151.25
    assert lineage["effective_ts_utc"] == "2026-01-01T10:00:30+00:00"
    assert lineage["event_time"] == "2026-01-01T10:00:30+00:00"
    assert lineage["timestamp_quality"] == "VALID"


def test_price_feature_preserves_context_effective_time_when_row_timestamp_is_invalid(mock_ctx):
    mock_ctx.effective_ts_utc = dt.datetime(2026, 1, 1, 10, 0, 45, tzinfo=dt.timezone.utc)
    mock_ctx.event_time_utc = mock_ctx.effective_ts_utc
    mock_ctx.effective_time_source = "event_time"
    mock_ctx.timestamp_quality = "VALID"
    mock_ctx.time_provenance_degraded = False

    payload = [{"close": 150.0, "t": "invalid_date_string"}]
    bundle = extract_price_features(payload, mock_ctx)

    lineage = bundle.meta["price"]["metric_lineage"]
    assert bundle.features["spot"] == 150.0
    assert lineage["effective_ts_utc"] == mock_ctx.effective_ts_utc.isoformat()
    assert lineage["timestamp_source"] == "event_time"
    assert lineage["timestamp_quality"] == "VALID"


def test_vol_skew_uses_latest_row_timestamp_when_available(mock_ctx):
    payload = {
        "history": [
            {"value": 0.11, "date": "2026-01-01T09:59:00+00:00"},
            {"value": 0.19, "date": "2026-01-01T10:00:30+00:00"},
        ]
    }
    bundle = extract_vol_skew(payload, mock_ctx)

    lineage = bundle.meta["skew"]["metric_lineage"]
    assert bundle.features["vol_skew"] == 0.19
    assert lineage["effective_ts_utc"] == "2026-01-01T10:00:30+00:00"
    assert bundle.meta["skew"]["details"]["selected_timestamp_key"] == "date"


def test_vol_skew_uses_context_degraded_effective_time_when_payload_is_timestamp_poor(mock_ctx):
    mock_ctx.effective_ts_utc = dt.datetime(2026, 1, 1, 10, 1, 0, tzinfo=dt.timezone.utc)
    mock_ctx.effective_time_source = "documented_asof_contemporaneous"
    mock_ctx.timestamp_quality = "DEGRADED"
    mock_ctx.time_provenance_degraded = True

    payload = {"history": [{"value": 0.13}]}
    bundle = extract_vol_skew(payload, mock_ctx)

    lineage = bundle.meta["skew"]["metric_lineage"]
    assert bundle.features["vol_skew"] == 0.13
    assert lineage["effective_ts_utc"] == mock_ctx.effective_ts_utc.isoformat()
    assert lineage["timestamp_source"] == "documented_asof_contemporaneous"
    assert lineage["timestamp_quality"] == "DEGRADED"


def test_dealer_greeks_snapshot_family_reclassifies_stale_row_timestamp_to_context(mock_ctx):
    mock_ctx.effective_ts_utc = dt.datetime(2026, 3, 5, 15, 0, tzinfo=dt.timezone.utc)
    mock_ctx.event_time_utc = mock_ctx.effective_ts_utc
    mock_ctx.effective_time_source = "documented_asof_contemporaneous"
    mock_ctx.timestamp_quality = "DEGRADED"
    mock_ctx.time_provenance_degraded = True
    mock_ctx.lagged = True
    mock_ctx.time_semantics_family = "greeks_snapshot"
    mock_ctx.max_trusted_source_age_seconds = 7200

    payload = [
        {
            "gamma_exposure": 1500000.0,
            "vanna_exposure": 250000.0,
            "charm_exposure": -125000.0,
            "date": "2026-03-05T03:00:00+00:00",
        }
    ]
    bundle = extract_dealer_greeks(payload, mock_ctx)
    lineage = bundle.meta["greeks"]["metric_lineage"]

    assert lineage["effective_ts_utc"] == mock_ctx.effective_ts_utc.isoformat()
    assert lineage["timestamp_source"] == "documented_asof_contemporaneous"
    assert lineage["time_provenance_degraded"] is True
    assert bundle.meta["greeks"]["details"]["reclassified_snapshot_family"] == "greeks_snapshot"


def test_gex_snapshot_family_reclassifies_stale_payload_timestamp_to_context(mock_ctx):
    mock_ctx.effective_ts_utc = dt.datetime(2026, 3, 5, 15, 0, tzinfo=dt.timezone.utc)
    mock_ctx.event_time_utc = mock_ctx.effective_ts_utc
    mock_ctx.effective_time_source = "documented_asof_contemporaneous"
    mock_ctx.timestamp_quality = "DEGRADED"
    mock_ctx.time_provenance_degraded = True
    mock_ctx.lagged = True
    mock_ctx.time_semantics_family = "gex_snapshot"
    mock_ctx.max_trusted_source_age_seconds = 7200

    payload = [
        {"gamma_exposure": 100.0, "date": "2026-03-05T04:00:00+00:00"},
        {"gamma_exposure": 50.0, "date": "2026-03-05T04:05:00+00:00"},
    ]
    bundle = extract_gex_sign(payload, mock_ctx)
    lineage = bundle.meta["gex"]["metric_lineage"]

    assert bundle.features["net_gex_sign"] == 1.0
    assert lineage["effective_ts_utc"] == mock_ctx.effective_ts_utc.isoformat()
    assert lineage["timestamp_source"] == "documented_asof_contemporaneous"
    assert bundle.meta["gex"]["details"]["reclassified_snapshot_family"] == "gex_snapshot"
