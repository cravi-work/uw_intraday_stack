from __future__ import annotations

from src.endpoint_truth import EndpointContext, EndpointPayloadClass, FreshnessState
from src.features import extract_dealer_greeks


def test_dealer_greeks_effective_ts_falls_back_to_payload_date_midnight_utc() -> None:
    ctx = EndpointContext(
        endpoint_id=1,
        method="GET",
        path="/api/stock/TSLA/greek-exposure",
        operation_id=None,
        signature="GET /api/stock/{ticker}/greek-exposure",
        used_event_id=None,
        payload_class=EndpointPayloadClass.SUCCESS_HAS_DATA,
        freshness_state=FreshnessState.FRESH,
        stale_age_min=None,
        na_reason=None,
        # No effective timestamp provided by the endpoint contract.
        effective_ts_utc=None,
        effective_time_source=None,
        timestamp_quality=None,
        time_provenance_degraded=False,
        endpoint_name="greek_exposure",
        endpoint_purpose="options_snapshot",
        decision_path=True,
        missing_affects_confidence=True,
        stale_affects_confidence=True,
        purpose_contract_version="v1",
    )

    payload = {
        "data": [
            {
                "date": "2026-03-06",
                "dealer_vanna": 1.0,
                "dealer_charm": 2.0,
                "net_gamma_exposure_notional": 100.0,
            }
        ]
    }

    bundle = extract_dealer_greeks(payload, ctx)
    meta = (bundle.meta or {}).get("greeks", {})
    eff = meta.get("metric_lineage", {}).get("effective_ts_utc")
    assert eff == "2026-03-06T00:00:00+00:00"
