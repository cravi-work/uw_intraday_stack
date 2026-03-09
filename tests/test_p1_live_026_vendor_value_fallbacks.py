import datetime as dt


from src.endpoint_truth import EndpointContext
from src.features import extract_gex_sign, extract_oi_features


ASOF_UTC = dt.datetime(2026, 3, 6, 20, 50, tzinfo=dt.timezone.utc)


def _ctx(path: str, endpoint_id: int = 1) -> EndpointContext:
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
        effective_ts_utc=ASOF_UTC,
        as_of_time_utc=ASOF_UTC,
        processed_at_utc=ASOF_UTC,
        received_at_utc=ASOF_UTC,
        effective_time_source="payload_effective_time",
        timestamp_quality="VALID",
        endpoint_name=f"endpoint_{endpoint_id}",
        endpoint_purpose="context",
        decision_path=False,
        missing_affects_confidence=True,
        stale_affects_confidence=True,
        purpose_contract_version="endpoint_purpose/v1",
    )


def test_gex_sign_accepts_single_row_dict_payload_and_camelcase_key() -> None:
    ctx = _ctx("/api/stock/TSLA/spot-exposures")
    payload = {"netGex": -100.0}

    out = extract_gex_sign(payload, ctx)
    assert out.features["net_gex_sign"] == -1


def test_oi_pressure_falls_back_to_summary_call_put_totals() -> None:
    ctx = _ctx("/api/stock/TSLA/oi")
    payload = {"call_open_interest": 200.0, "put_open_interest": 100.0}

    out = extract_oi_features(payload, ctx)
    assert out.features["oi_pressure"] == (200.0 - 100.0) / (200.0 + 100.0)
    assert out.meta["oi"]["details"]["status"] == "computed_from_summary_totals"
