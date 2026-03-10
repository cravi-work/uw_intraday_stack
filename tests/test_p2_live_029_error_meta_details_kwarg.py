import datetime

from src.endpoint_truth import EndpointContext
from src.features import extract_gex_sign


def _ctx(path: str = "/api/stock/TSLA/spot-exposures", effective_ts: str = "2026-03-06T20:30:00+00:00") -> EndpointContext:
    ts = datetime.datetime.fromisoformat(effective_ts)
    return EndpointContext(
        endpoint_id=1,
        method="GET",
        path=path,
        operation_id=None,
        signature="sig",
        used_event_id=None,
        payload_class="OK",
        freshness_state="FRESH",
        stale_age_min=None,
        na_reason=None,
        endpoint_asof_ts_utc=ts,
        effective_ts_utc=ts,
        as_of_time_utc=ts,
        processed_at_utc=ts,
        received_at_utc=ts,
        effective_time_source="payload.timestamp",
        timestamp_quality="GOOD",
        endpoint_name="spot_exposures",
        endpoint_purpose="decision",
        decision_path=True,
        missing_affects_confidence=True,
        stale_affects_confidence=True,
        purpose_contract_version="p1-001",
    )


def test_extract_gex_sign_missing_gamma_fields_does_not_crash_and_includes_details():
    ctx = _ctx()
    # Payload with rows but no known gamma exposure fields.
    payload = [{"date": "2026-03-06", "foo": 1}]

    out = extract_gex_sign(payload, ctx)

    assert out.features["net_gex_sign"] is None
    assert "gex" in out.meta
    meta = out.meta["gex"]

    # This is the exact failure mode we saw in live logs (details kwarg to _build_error_meta).
    assert meta.get("na_reason") == "missing_gamma_exposure_fields"

    details = meta.get("details") or {}
    assert "attempted_field_aliases" in details
    assert "sample_row_keys" in details
