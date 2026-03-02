import math
from unittest.mock import MagicMock

import pytest

from src.endpoint_truth import EndpointContext
from src.features import (
    extract_darkpool_pressure,
    extract_litflow_pressure,
    extract_oi_features,
    extract_volatility_features,
)


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
    ctx.received_at_utc = None
    ctx.processed_at_utc = None
    ctx.as_of_time_utc = None
    ctx.source_revision = None
    ctx.effective_time_source = "missing_provider_time"
    ctx.timestamp_quality = "DEGRADED"
    ctx.lagged = False
    ctx.time_provenance_degraded = True
    return ctx


def test_darkpool_pressure_directionless_totals_are_suppressed(mock_ctx):
    payload = [
        {"price": 100.0, "volume": 500},
        {"price": 100.0, "volume": float("nan")},
        {"price": float("inf"), "volume": 100},
    ]
    bundle = extract_darkpool_pressure(payload, mock_ctx)

    assert bundle.features["darkpool_pressure"] is None
    assert bundle.meta["darkpool"]["na_reason"] == "directionless_total"
    assert bundle.meta["darkpool"]["details"]["status"] == "suppressed_directionless_darkpool_total"


def test_litflow_pressure_side_logic_is_bounded(mock_ctx):
    payload = [
        {"price": 10.0, "size": 100, "side": "ASK"},
        {"price": 10.0, "size": 50, "side": "BID"},
    ]
    bundle = extract_litflow_pressure(payload, mock_ctx)
    val = bundle.features["litflow_pressure"]

    assert val == pytest.approx((1000.0 - 500.0) / 1500.0)
    assert -1.0 <= val <= 1.0
    assert math.isfinite(val)


def test_oi_pressure_requires_directional_contract_semantics(mock_ctx):
    payload = [
        {"strike": 100.0, "open_interest": 1_000_000_000.0},
        {"strike": 105.0, "open_interest": 2_000_000_000.0},
    ]
    bundle = extract_oi_features(payload, mock_ctx)

    assert bundle.features["oi_pressure"] is None
    assert bundle.meta["oi"]["na_reason"] == "missing_put_call_or_directional_rows"
    assert bundle.meta["oi"]["details"]["status"] == "suppressed_directionless_oi_total"


def test_oi_pressure_signed_and_bounded_when_put_call_available(mock_ctx):
    payload = [
        {"strike": 95.0, "open_interest": 1200.0, "put_call": "CALL", "spot": 100.0},
        {"strike": 105.0, "open_interest": 800.0, "put_call": "PUT", "spot": 100.0},
        {"strike": 100.0, "open_interest": 1000.0, "put_call": "CALL", "spot": 100.0},
    ]
    bundle = extract_oi_features(payload, mock_ctx)
    val = bundle.features["oi_pressure"]

    assert val is not None
    assert -1.0 <= val <= 1.0
    assert math.isfinite(val)
    assert bundle.meta["oi"]["metric_lineage"]["units_expected"] == "Directional Imbalance Ratio [-1, 1]"


def test_volatility_finite_guards(mock_ctx):
    payload = [{"iv_rank": float("nan")}]
    bundle = extract_volatility_features(payload, mock_ctx)

    assert bundle.features["iv_rank"] is None
