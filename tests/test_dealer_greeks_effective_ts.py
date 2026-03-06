import datetime
from unittest.mock import MagicMock

from src.endpoint_truth import FreshnessState
from src.features import extract_dealer_greeks


def _mock_ctx(effective_ts_utc: datetime.datetime):
    ctx = MagicMock()
    ctx.endpoint_id = "greek_exposure"
    ctx.method = "GET"
    ctx.path = "/api/stock/SPY/greek-exposure"
    ctx.operation_id = "greek_exposure"
    ctx.signature = "GET /api/stock/{ticker}/greek-exposure"
    ctx.payload_class = "json"
    ctx.used_event_id = None
    ctx.freshness_state = FreshnessState.FRESH
    ctx.stale_age_min = 0.0
    ctx.na_reason = None

    ctx.effective_ts_utc = effective_ts_utc
    ctx.event_time_utc = None
    ctx.source_publish_time_utc = None
    ctx.as_of_time_utc = effective_ts_utc
    return ctx


def test_dealer_greeks_effective_ts_uses_endpoint_truth_not_row_date_midnight():
    # Date-only greeks rows (YYYY-MM-DD) are *not* precise event timestamps.
    # Effective time should come from EndpointTruth (ctx.effective_ts_utc) to avoid
    # midnight-UTC join-skew violations after market open/close.
    asof = datetime.datetime(2026, 3, 5, 21, 24, tzinfo=datetime.timezone.utc)
    ctx = _mock_ctx(asof)

    greek_payload = [
        {
            "underlying": "SPY",
            "expiration": "2026-03-20",
            "strike": 100.0,
            "put_call": "CALL",
            "multiplier": 100,
            "deliverable_shares": 100,
            "vanna": 0.5,
            "charm": 0.25,
            "gamma_exposure": 1_500_000_000,
            "date": "2026-03-05",
        }
    ]

    bundle = extract_dealer_greeks(greek_payload, ctx)
    meta = bundle.meta["greeks"]

    assert meta["metric_lineage"]["effective_ts_utc"] == asof.isoformat()
    assert meta["metric_lineage"]["event_time"] is None

    details = meta["details"]
    assert details["payload_row_date_raw"] == "2026-03-05"
    # The exact time-of-day can vary by local timezone when the source is date-only,
    # but it must remain traceable to the same date marker.
    assert details["payload_row_date_utc"].startswith("2026-03-05T")
