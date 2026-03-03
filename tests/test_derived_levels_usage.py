import datetime as dt

import pytest

from src.analytics import (
    build_darkpool_levels,
    build_gex_levels,
    build_oi_walls,
    derived_level_usage_contract,
    is_prediction_consumed_derived_level,
    reporting_only_derived_level_types,
)
from src.endpoint_truth import EndpointContext
from src.features import extract_all
from src.ingest_engine import generate_predictions
from src.models import SessionState


ASOF_UTC = dt.datetime(2026, 1, 15, 15, 0, tzinfo=dt.timezone.utc)


@pytest.mark.parametrize(
    ("builder", "payload", "kwargs"),
    [
        (
            build_gex_levels,
            [
                {"strike": 100.0, "gamma_exposure": -500.0},
                {"strike": 105.0, "gamma_exposure": 250.0},
            ],
            {},
        ),
        (
            build_oi_walls,
            [
                {"strike": 150.0, "open_interest": 1000.0, "put_call": "CALL", "spot": 150.0},
                {"strike": 145.0, "open_interest": 900.0, "put_call": "PUT", "spot": 150.0},
            ],
            {"spot": 150.0},
        ),
        (
            build_darkpool_levels,
            [
                {"price": 150.00, "volume": 10000.0},
                {"price": 150.01, "volume": 2500.0},
            ],
            {},
        ),
    ],
)
def test_derived_level_builders_explicitly_mark_outputs_report_only(builder, payload, kwargs):
    levels = builder(payload, **kwargs)
    assert levels, "expected builder to emit at least one derived level"

    known_types = set(reporting_only_derived_level_types())

    for level_type, _, _, details in levels:
        contract = details["derived_level_contract"]
        assert level_type in known_types
        assert contract == derived_level_usage_contract(level_type)
        assert contract["decision_path_role"] == "report-only"
        assert contract["prediction_consumed"] is False
        assert contract["model_input_eligible"] is False
        assert contract["feature_contract_state"] == "DEMOTED"
        assert is_prediction_consumed_derived_level(level_type) is False


def _ctx(endpoint_id: int, path: str, effective_ts: dt.datetime) -> EndpointContext:
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
        endpoint_asof_ts_utc=effective_ts,
        effective_ts_utc=effective_ts,
        as_of_time_utc=effective_ts,
        processed_at_utc=effective_ts,
        received_at_utc=effective_ts,
        effective_time_source="endpoint_provenance",
        timestamp_quality="VALID",
    )


def _prediction_cfg():
    return {
        "ingestion": {"cadence_minutes": 5},
        "model": {
            "model_name": "phase0_additive",
            "model_version": "2.0.0",
            "target_spec": {
                "target_name": "intraday_direction_3class",
                "target_version": "runtime_v2",
            },
            "calibration": {
                "artifact_name": "phase0_additive_calibration",
                "artifact_version": "cal_v1",
                "bins": [0.0, 0.5, 1.0],
                "mapped": [0.0, 0.5, 1.0],
            },
        },
        "validation": {
            "invalid_after_minutes": 60,
            "tolerance_minutes": 10,
            "max_horizon_drift_minutes": 10,
            "flat_threshold_pct": 0.001,
            "fallback_max_age_minutes": 15,
            "alignment_tolerance_sec": 900,
            "use_default_required_features": False,
            "emit_to_close_horizon": False,
            "horizons_minutes": [15],
            "horizon_weights_source": "explicit",
            "horizon_critical_features": {"15": ["spot"]},
            "horizon_weights": {"15": {"spot": 0.001, "net_gex_sign": 0.5}},
            "label_contract": {
                "label_version": "runtime_v1",
                "threshold_policy_version": "runtime_v1",
            },
        },
    }


def test_extract_all_preserves_report_only_usage_metadata_for_levels():
    payloads = {
        1: [{"close": 150.0, "t": ASOF_UTC.timestamp()}],
        2: [
            {"strike": 100.0, "gamma_exposure": -500.0},
            {"strike": 105.0, "gamma_exposure": 250.0},
        ],
    }
    contexts = {
        1: _ctx(1, "/api/stock/{ticker}/ohlc/{candle_size}", ASOF_UTC),
        2: _ctx(2, "/api/stock/{ticker}/spot-exposures", ASOF_UTC),
    }

    f_rows, l_rows = extract_all(payloads, contexts)

    assert any(f["feature_key"] == "spot" for f in f_rows)
    assert l_rows, "expected derived levels to remain available for reporting"

    feature_keys = {f["feature_key"] for f in f_rows}
    level_types = {l["level_type"] for l in l_rows}
    assert level_types.isdisjoint(feature_keys)

    for level in l_rows:
        contract = level["meta_json"]["level_usage_contract"]
        assert contract["decision_path_role"] == "report-only"
        assert contract["prediction_consumed"] is False
        assert level["meta_json"]["metric_lineage"]["decision_path_role"] == "report-only"
        assert level["meta_json"]["metric_lineage"]["feature_contract_state"] == "DEMOTED"


def test_prediction_path_explicitly_excludes_report_only_levels():
    payloads = {
        1: [{"close": 150.0, "t": ASOF_UTC.timestamp()}],
        2: [
            {"strike": 100.0, "gamma_exposure": -500.0},
            {"strike": 105.0, "gamma_exposure": 250.0},
        ],
    }
    contexts = {
        1: _ctx(1, "/api/stock/{ticker}/ohlc/{candle_size}", ASOF_UTC),
        2: _ctx(2, "/api/stock/{ticker}/spot-exposures", ASOF_UTC),
    }
    valid_features, levels = extract_all(payloads, contexts)
    predictions = generate_predictions(
        _prediction_cfg(),
        snapshot_id=1,
        valid_features=valid_features,
        asof_utc=ASOF_UTC,
        session_enum=SessionState.RTH,
        sec_to_close=3600,
        endpoint_coverage=1.0,
    )

    assert levels, "reporting path should still surface derived levels"
    assert predictions, "prediction path should still operate on true feature inputs"

    level_types = {level["level_type"] for level in levels}
    pred = next(p for p in predictions if p["horizon_minutes"] == 15)

    feature_policy_keys = set(pred["meta_json"]["freshness_registry_diagnostics"]["feature_policies"].keys())
    critical_feature_keys = set(pred["meta_json"]["horizon_contract"]["resolved_critical_features"])

    assert level_types.isdisjoint(feature_policy_keys)
    assert level_types.isdisjoint(critical_feature_keys)
    assert pred["prob_up"] is not None
    assert pred["prob_down"] is not None
    assert pred["prob_flat"] is not None
