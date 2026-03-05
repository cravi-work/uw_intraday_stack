import datetime as dt

from src.endpoint_truth import EndpointContext
from src.features import (
    extract_dealer_greeks,
    extract_gex_sign,
    extract_litflow_pressure,
    extract_oi_features,
    extract_smart_whale_pressure,
    extract_volatility_features,
)

ASOF_UTC = dt.datetime(2026, 3, 2, 15, 30, tzinfo=dt.timezone.utc)


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
        effective_time_source="endpoint_provenance",
        timestamp_quality="VALID",
        endpoint_name=f"endpoint_{endpoint_id}",
        endpoint_purpose="signal-critical",
        decision_path=True,
        missing_affects_confidence=True,
        stale_affects_confidence=True,
        purpose_contract_version="endpoint_purpose/v1",
    )


def _assert_bounded_contract(
    lineage: dict,
    *,
    emitted_units: str,
    raw_input_units: str,
    lower: float,
    upper: float,
    output_domain: str = "closed_interval",
) -> None:
    assert lineage["decision_eligible"] is True
    assert lineage["bounded_output"] is True
    assert lineage["output_domain_contract_version"] == "output_domain/v1"
    assert lineage["output_domain"] == output_domain
    assert lineage["emitted_units"] == emitted_units
    assert lineage["raw_input_units"] == raw_input_units
    assert lineage["expected_bounds"] == {"lower": float(lower), "upper": float(upper), "inclusive": True}



def test_smart_whale_pressure_publishes_emitted_domain_contract():
    payload = [
        {
            "underlying": "AAPL",
            "expiration": "2026-06-19",
            "strike": 150.0,
            "put_call": "CALL",
            "premium": 60_000.0,
            "dte": 5.0,
            "side": "BUY",
            "multiplier": 100.0,
            "deliverable_shares": 100.0,
        }
    ]

    bundle = extract_smart_whale_pressure(payload, _ctx("/api/stock/{ticker}/flow-recent"))

    assert bundle.features["smart_whale_pressure"] is not None
    lineage = bundle.meta["flow"]["metric_lineage"]
    _assert_bounded_contract(
        lineage,
        emitted_units="normalized_directional_pressure",
        raw_input_units="Net Premium Flow (USD)",
        lower=-1.0,
        upper=1.0,
    )
    assert lineage["units_expected"] == "Normalized Directional Pressure [-1, 1]"



def test_dealer_greeks_publish_normalized_output_domain_contract():
    payload = [
        {
            "underlying": "AAPL",
            "expiration": "2026-06-19",
            "strike": 150.0,
            "put_call": "CALL",
            "date": "2026-03-02T15:30:00Z",
            "gamma_exposure": 1_500_000.0,
            "vanna_exposure": 120_000.0,
            "charm_exposure": 80_000.0,
            "multiplier": 100.0,
            "deliverable_shares": 100.0,
        }
    ]

    bundle = extract_dealer_greeks(payload, _ctx("/api/stock/{ticker}/greek-exposure"))

    assert bundle.features["dealer_vanna"] is not None
    assert bundle.features["dealer_charm"] is not None
    assert bundle.features["net_gamma_exposure_notional"] is not None
    lineage = bundle.meta["greeks"]["metric_lineage"]
    _assert_bounded_contract(
        lineage,
        emitted_units="normalized_signed_exposure",
        raw_input_units="Notional Exposure (USD)",
        lower=-1.0,
        upper=1.0,
    )
    assert lineage["units_expected"] == "Normalized Signed Exposure [-1, 1]"



def test_other_bounded_decision_features_publish_machine_readable_bounds():
    oi_bundle = extract_oi_features(
        [
            {
                "underlying": "AAPL",
                "expiration": "2026-06-19",
                "strike": 145.0,
                "open_interest": 1200.0,
                "put_call": "CALL",
                "spot": 150.0,
                "multiplier": 100.0,
                "deliverable_shares": 100.0,
            },
            {
                "underlying": "AAPL",
                "expiration": "2026-06-19",
                "strike": 155.0,
                "open_interest": 900.0,
                "put_call": "PUT",
                "spot": 150.0,
                "multiplier": 100.0,
                "deliverable_shares": 100.0,
            },
        ],
        _ctx("/api/stock/{ticker}/oi-per-strike"),
    )
    oi_lineage = oi_bundle.meta["oi"]["metric_lineage"]
    _assert_bounded_contract(
        oi_lineage,
        emitted_units="directional_imbalance_ratio",
        raw_input_units="Open Interest (contracts)",
        lower=-1.0,
        upper=1.0,
    )

    litflow_bundle = extract_litflow_pressure(
        [
            {"price": 10.0, "size": 100, "side": "ASK"},
            {"price": 10.0, "size": 80, "side": "BID"},
        ],
        _ctx("/api/lit-flow/{ticker}"),
    )
    litflow_lineage = litflow_bundle.meta["litflow"]["metric_lineage"]
    _assert_bounded_contract(
        litflow_lineage,
        emitted_units="directional_imbalance_ratio",
        raw_input_units="Notional Flow (USD)",
        lower=-1.0,
        upper=1.0,
    )

    iv_rank_bundle = extract_volatility_features(
        [{"iv_rank": 0.62}],
        _ctx("/api/stock/{ticker}/iv-rank"),
    )
    iv_rank_lineage = iv_rank_bundle.meta["vol"]["metric_lineage"]
    _assert_bounded_contract(
        iv_rank_lineage,
        emitted_units="percentile_rank",
        raw_input_units="Percentile Rank",
        lower=0.0,
        upper=1.0,
    )

    gex_bundle = extract_gex_sign(
        [{"gamma_exposure": 1200.0}, {"gamma_exposure": -200.0}],
        _ctx("/api/stock/{ticker}/spot-exposures"),
    )
    gex_lineage = gex_bundle.meta["gex"]["metric_lineage"]
    _assert_bounded_contract(
        gex_lineage,
        emitted_units="directional_sign",
        raw_input_units="Gamma Exposure (provider aggregate units)",
        lower=-1.0,
        upper=1.0,
        output_domain="discrete_sign",
    )
    assert gex_lineage["allowed_values"] == [-1.0, 0.0, 1.0]
