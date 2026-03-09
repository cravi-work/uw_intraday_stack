import datetime as dt


from src.endpoint_truth import EndpointContext
from src.features import extract_dealer_greeks, extract_vol_term_structure


ASOF_UTC = dt.datetime(2026, 3, 6, 20, 50, tzinfo=dt.timezone.utc)


def _ctx_missing_effective_ts(path: str, endpoint_id: int = 1) -> EndpointContext:
    """EndpointContext with no inferred effective timestamp (simulates date-only snapshot payloads)."""
    return EndpointContext(
        endpoint_id=endpoint_id,
        method="GET",
        path=path,
        operation_id="test_op",
        signature=f"GET {path}",
        used_event_id=f"event_{endpoint_id}",
        payload_class="SUCCESS_HAS_DATA",
        freshness_state="FRESH",
        stale_age_min=0,
        na_reason=None,
        endpoint_asof_ts_utc=ASOF_UTC,
        effective_ts_utc=None,
        as_of_time_utc=ASOF_UTC,
        processed_at_utc=ASOF_UTC,
        received_at_utc=ASOF_UTC,
        effective_time_source="unavailable",
        timestamp_quality="MISSING",
        endpoint_name=f"endpoint_{endpoint_id}",
        endpoint_purpose="context",
        decision_path=False,
        missing_affects_confidence=True,
        stale_affects_confidence=True,
        purpose_contract_version="endpoint_purpose/v1",
    )


def test_dealer_greeks_date_only_payload_defaults_effective_ts_to_midnight_utc() -> None:
    payload = [
        {
            "date": "2026-03-06",
            "dealer_vanna": 1.0e9,
            "dealer_charm": -2.0e9,
            "net_gamma_exposure_notional": 3.0e9,
        }
    ]
    ctx = _ctx_missing_effective_ts("/api/stock/TSLA/greek-exposure")

    out = extract_dealer_greeks(payload, ctx)
    lineage = out.meta["greeks"]["metric_lineage"]

    assert lineage["effective_ts_utc"] == "2026-03-06T00:00:00+00:00"
    assert lineage["timestamp_source"] == "payload_date_midnight_utc"
    assert lineage["timestamp_quality"] == "DATE_ONLY"
    assert lineage["time_provenance_degraded"] is True


def test_vol_term_structure_date_only_payload_defaults_effective_ts_to_midnight_utc() -> None:
    payload = [
        {"date": "2026-03-06", "dte": 7, "iv": 0.35},
        {"date": "2026-03-06", "dte": 30, "iv": 0.40},
    ]
    ctx = _ctx_missing_effective_ts("/api/stock/TSLA/term-structure")

    out = extract_vol_term_structure(payload, ctx)
    lineage = out.meta["vol_ts"]["metric_lineage"]

    assert lineage["effective_ts_utc"] == "2026-03-06T00:00:00+00:00"
    assert lineage["timestamp_source"] == "payload_date_midnight_utc"
    assert lineage["timestamp_quality"] == "DATE_ONLY"
    assert lineage["time_provenance_degraded"] is True
