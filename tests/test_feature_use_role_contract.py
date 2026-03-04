import datetime as dt

from src.endpoint_truth import EndpointContext
from src.features import extract_all


ASOF_UTC = dt.datetime(2026, 1, 15, 15, 0, tzinfo=dt.timezone.utc)


def _ctx(
    endpoint_id: int,
    path: str,
    *,
    endpoint_purpose: str,
    decision_path: bool,
    missing_affects_confidence: bool,
    stale_affects_confidence: bool,
) -> EndpointContext:
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
        effective_time_source="endpoint_provenance",
        timestamp_quality="VALID",
        endpoint_name=f"endpoint_{endpoint_id}",
        endpoint_purpose=endpoint_purpose,
        decision_path=decision_path,
        missing_affects_confidence=missing_affects_confidence,
        stale_affects_confidence=stale_affects_confidence,
        purpose_contract_version="endpoint_purpose/v1",
    )


def test_signal_critical_feature_is_marked_decision_eligible():
    payloads = {1: [{"close": 150.0, "t": ASOF_UTC.timestamp()}]}
    contexts = {
        1: _ctx(
            1,
            "/api/stock/{ticker}/ohlc/{candle_size}",
            endpoint_purpose="signal-critical",
            decision_path=True,
            missing_affects_confidence=True,
            stale_affects_confidence=True,
        )
    }

    f_rows, _ = extract_all(payloads, contexts)
    spot = next(f for f in f_rows if f["feature_key"] == "spot")
    meta = spot["meta_json"]

    assert meta["use_role"] == "signal-critical"
    assert meta["decision_eligible"] is True
    assert meta["missing_affects_confidence"] is True
    assert meta["stale_affects_confidence"] is True
    assert meta["feature_use_contract"]["contract_version"] == "endpoint_purpose/v1"
    assert meta["feature_use_contract"]["contract_source"] == "endpoint_context"
    assert meta["metric_lineage"]["decision_path_role"] == "signal-critical"
    assert meta["metric_lineage"]["decision_eligible"] is True
    assert meta["source_endpoints"][0]["purpose"] == "signal-critical"
    assert meta["source_endpoints"][0]["decision_path"] is True


def test_report_only_feature_remains_extractable_but_non_decision_eligible():
    payloads = {
        1: [
            {"price": 150.0, "volume": 1000.0, "side": "BUY"},
            {"price": 149.5, "volume": 800.0, "side": "SELL"},
        ]
    }
    contexts = {
        1: _ctx(
            1,
            "/api/darkpool/{ticker}",
            endpoint_purpose="report-only",
            decision_path=False,
            missing_affects_confidence=False,
            stale_affects_confidence=False,
        )
    }

    f_rows, _ = extract_all(payloads, contexts)
    darkpool = next(f for f in f_rows if f["feature_key"] == "darkpool_pressure")
    meta = darkpool["meta_json"]

    assert darkpool["feature_value"] is not None
    assert meta["use_role"] == "report-only"
    assert meta["decision_eligible"] is False
    assert meta["missing_affects_confidence"] is False
    assert meta["stale_affects_confidence"] is False
    assert meta["metric_lineage"]["decision_path_role"] == "report-only"
    assert meta["metric_lineage"]["decision_eligible"] is False
    assert meta["source_endpoints"][0]["purpose"] == "report-only"
    assert meta["source_endpoints"][0]["decision_path"] is False
