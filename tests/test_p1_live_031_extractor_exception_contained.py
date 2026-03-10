import pytest

from src.endpoint_truth import EndpointContext
import src.features as features


def test_extract_all_contains_extractor_exceptions(monkeypatch: pytest.MonkeyPatch) -> None:
    # Force the GEX extractor to raise so we can verify the exception is contained
    # and converted into a safe NA feature row rather than crashing the whole ingest.
    def _boom(payload, ctx):
        raise RuntimeError("boom")

    monkeypatch.setattr(features, "extract_gex_sign", _boom)

    ctx = EndpointContext(
        endpoint_id=1,
        method="GET",
        path="/api/stock/{ticker}/spot-exposures",
        operation_id="spot_exposures",
        signature="GET /api/stock/{ticker}/spot-exposures",
        used_event_id=None,
        payload_class="json",
        freshness_state="FRESH",
        stale_age_min=0,
        na_reason=None,
    )

    f_rows, l_rows = features.extract_all({1: {}}, {1: ctx})

    # We should emit a net_gex_sign row (value=None) with a structured NA reason.
    gex_rows = [r for r in f_rows if r["feature_key"] == "net_gex_sign"]
    assert len(gex_rows) == 1
    assert gex_rows[0]["feature_value"] is None
    assert gex_rows[0]["meta_json"].get("na_reason") == "extractor_exception"
