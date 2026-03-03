import pytest
from unittest.mock import MagicMock

from src.analytics import build_oi_walls
from src.endpoint_truth import EndpointContext
from src.features import (
    extract_dealer_greeks,
    extract_oi_features,
    extract_smart_whale_pressure,
)
from src.instruments import normalize_option_rows


def _mock_ctx(path: str = "/api/stock/{ticker}/oi-per-strike"):
    ctx = MagicMock(spec=EndpointContext)
    ctx.freshness_state = "FRESH"
    ctx.na_reason = None
    ctx.path = path
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


def test_adjusted_contract_identity_is_canonicalized():
    rows = [
        {
            "underlying": "AAPL",
            "expiration": "2026-06-19",
            "strike": 150.0,
            "put_call": "CALL",
            "multiplier": 150.0,
            "deliverable_shares": 150.0,
            "adjusted": True,
        }
    ]

    summary = normalize_option_rows(rows)

    assert summary.status == "NORMALIZED"
    ident = summary.normalized_rows[0].identity
    assert ident.adjustment_flag is True
    assert ident.multiplier == pytest.approx(150.0)
    assert ident.deliverable_shares == pytest.approx(150.0)
    assert ident.canonical_contract_key.endswith("mult=150|deliverable=shares:150|adj=1")


def test_adjusted_contract_oi_walls_use_normalized_deliverable_units():
    payload = [
        {
            "underlying": "AAPL",
            "expiration": "2026-06-19",
            "strike": 145.0,
            "open_interest": 10.0,
            "put_call": "PUT",
            "multiplier": 150.0,
            "deliverable_shares": 150.0,
            "adjusted": True,
        },
        {
            "underlying": "AAPL",
            "expiration": "2026-06-19",
            "strike": 155.0,
            "open_interest": 8.0,
            "put_call": "CALL",
            "multiplier": 150.0,
            "deliverable_shares": 150.0,
            "adjusted": True,
        },
    ]

    levels = build_oi_walls(payload, spot=150.0)

    call_wall = next(level for level in levels if level[0] == "CALL_WALL")
    put_wall = next(level for level in levels if level[0] == "PUT_WALL")
    assert call_wall[2] == pytest.approx(8.0 * 150.0)
    assert put_wall[2] == pytest.approx(10.0 * 150.0)
    assert call_wall[3]["normalized_contract_rows"] == 2
    assert call_wall[3]["contract_normalization"]["status"] == "NORMALIZED"


def test_multiplier_mismatch_invalidates_oi_pressure():
    ctx = _mock_ctx()
    payload = [
        {
            "underlying": "AAPL",
            "expiration": "2026-06-19",
            "strike": 150.0,
            "open_interest": 10.0,
            "put_call": "CALL",
            "multiplier": 100.0,
            "deliverable_shares": 100.0,
        },
        {
            "underlying": "AAPL",
            "expiration": "2026-06-19",
            "strike": 150.0,
            "open_interest": 10.0,
            "put_call": "CALL",
            "multiplier": 150.0,
            "deliverable_shares": 150.0,
        },
    ]

    bundle = extract_oi_features(payload, ctx)

    assert bundle.features["oi_pressure"] is None
    assert bundle.meta["oi"]["na_reason"] == "contract_multiplier_conflict"
    assert bundle.meta["oi"]["details"]["contract_normalization"]["status"] == "INVALID"
    assert bundle.meta["oi"]["details"]["contract_normalization"]["series_conflicts"]


def test_unparseable_display_symbol_suppresses_flow_metric():
    ctx = _mock_ctx("/api/stock/{ticker}/flow-recent")
    payload = [
        {
            "option_symbol": "BADDISPLAY",
            "premium": 50000.0,
            "dte": 5.0,
            "side": "BUY",
            "put_call": "CALL",
        }
    ]

    bundle = extract_smart_whale_pressure(payload, ctx)

    assert bundle.features["smart_whale_pressure"] is None
    assert bundle.meta["flow"]["na_reason"] == "display_symbol_only_identity"
    assert bundle.meta["flow"]["details"]["contract_normalization"]["status"] == "INVALID"


def test_unparseable_display_symbol_suppresses_greek_metric():
    ctx = _mock_ctx("/api/stock/{ticker}/greek-exposure")
    payload = [
        {
            "option_symbol": "BADDISPLAY",
            "date": "2026-03-02T15:30:00Z",
            "gamma_exposure": 1000.0,
            "vanna_exposure": 100.0,
            "charm_exposure": 50.0,
        }
    ]

    bundle = extract_dealer_greeks(payload, ctx)

    assert bundle.features["dealer_vanna"] is None
    assert bundle.features["dealer_charm"] is None
    assert bundle.features["net_gamma_exposure_notional"] is None
    assert bundle.meta["greeks"]["na_reason"] == "display_symbol_only_identity"
    assert bundle.meta["greeks"]["details"]["contract_normalization"]["status"] == "INVALID"



def test_missing_multiplier_suppresses_contract_level_oi_metric():
    ctx = _mock_ctx()
    payload = [
        {
            "underlying": "AAPL",
            "expiration": "2026-06-19",
            "strike": 150.0,
            "put_call": "CALL",
            "open_interest": 10.0,
        }
    ]

    bundle = extract_oi_features(payload, ctx)

    assert bundle.features["oi_pressure"] is None
    assert bundle.meta["oi"]["na_reason"] == "missing_multiplier"
    assert bundle.meta["oi"]["details"]["contract_normalization"]["status"] == "INVALID"



def test_unparsed_deliverable_suppresses_contract_level_flow_metric():
    ctx = _mock_ctx("/api/stock/{ticker}/flow-recent")
    payload = [
        {
            "underlying": "AAPL",
            "expiration": "2026-06-19",
            "strike": 150.0,
            "put_call": "CALL",
            "premium": 50000.0,
            "dte": 5.0,
            "side": "BUY",
            "multiplier": 100.0,
            "deliverable": "cash + rights",
        }
    ]

    bundle = extract_smart_whale_pressure(payload, ctx)

    assert bundle.features["smart_whale_pressure"] is None
    assert bundle.meta["flow"]["na_reason"] == "unparsed_deliverable"
    assert bundle.meta["flow"]["details"]["contract_normalization"]["status"] == "INVALID"
