import datetime as dt
import logging

from src.ingest_engine import generate_predictions
from src.models import SessionState


def _cfg(weights: dict[str, float], criticals: list[str]) -> dict:
    return {
        "ingestion": {"cadence_minutes": 5},
        "validation": {
            "horizon_weights_source": "explicit",
            "horizons_minutes": [5],
            "horizon_weights": {"5": dict(weights)},
            "horizon_critical_features": {"5": list(criticals)},
            "use_default_required_features": False,
            "emit_to_close_horizon": False,
            "flat_threshold_pct": 0.001,
            "alignment_tolerance_sec": 900,
            "invalid_after_minutes": 60,
        },
        "model": {
            "model_name": "bounded_additive_score",
            "model_version": "test.v1",
            "confidence_cap": 0.55,
            "min_confidence": 0.35,
            "neutral_threshold": 0.55,
            "direction_margin": 0.08,
            "min_flat_prob": 0.15,
            "max_flat_prob": 0.65,
            "flat_from_data_quality_scale": 0.9,
            "target_spec": {
                "target_name": "intraday_direction_3class",
                "target_version": "test.v1",
            },
            "calibration": {
                "artifact_name": "test_calibration",
                "artifact_version": "test.cal.v1",
                "bins": [0.0, 0.5, 1.0],
                "mapped": [0.05, 0.5, 0.95],
            },
        },
    }


def _feature(
    feature_key: str,
    value: float | None,
    *,
    asof_utc: dt.datetime,
    path: str,
    use_role: str = "signal-critical",
    decision_eligible: bool | None = None,
    freshness_state: str = "FRESH",
    stale_age_min: int = 0,
) -> dict:
    if decision_eligible is None:
        decision_eligible = use_role == "signal-critical"
    contract = {
        "contract_version": "feature_use/v1",
        "use_role": use_role,
        "decision_path": bool(decision_eligible),
        "decision_eligible": bool(decision_eligible),
        "missing_affects_confidence": bool(decision_eligible),
        "stale_affects_confidence": bool(decision_eligible),
    }
    return {
        "feature_key": feature_key,
        "feature_value": value,
        "meta_json": {
            "source_endpoints": [
                {
                    "method": "GET",
                    "path": path,
                    "purpose": use_role,
                    "decision_path": bool(decision_eligible),
                    "missing_affects_confidence": bool(decision_eligible),
                    "stale_affects_confidence": bool(decision_eligible),
                    "purpose_contract_version": "feature_use/v1",
                }
            ],
            "freshness_state": freshness_state,
            "stale_age_min": stale_age_min,
            "na_reason": None,
            "feature_use_contract": contract,
            "use_role": use_role,
            "decision_eligible": bool(decision_eligible),
            "missing_affects_confidence": bool(decision_eligible),
            "stale_affects_confidence": bool(decision_eligible),
            "metric_lineage": {
                "effective_ts_utc": (asof_utc - dt.timedelta(minutes=1)).isoformat(),
                "source_path": path,
                "decision_path_role": use_role,
                "feature_use_contract_version": "feature_use/v1",
            },
            "details": {},
        },
    }


def _run(features: list[dict], cfg: dict) -> dict:
    predictions = generate_predictions(
        cfg,
        snapshot_id=123,
        valid_features=features,
        asof_utc=dt.datetime(2026, 3, 3, 15, 0, tzinfo=dt.timezone.utc),
        session_enum=SessionState.RTH,
        sec_to_close=None,
        endpoint_coverage=1.0,
    )
    assert len(predictions) == 1
    return predictions[0]


def test_report_only_feature_presence_or_absence_does_not_change_decision_outcome():
    asof_utc = dt.datetime(2026, 3, 3, 15, 0, tzinfo=dt.timezone.utc)
    cfg = _cfg(weights={"oi_pressure": 1.0, "darkpool_pressure": 0.0}, criticals=["spot"])
    base_features = [
        _feature("spot", 150.0, asof_utc=asof_utc, path="/api/stock/{ticker}/ohlc/{candle_size}"),
        _feature("oi_pressure", 0.8, asof_utc=asof_utc, path="/api/stock/{ticker}/oi-per-strike"),
    ]
    report_only_darkpool = _feature(
        "darkpool_pressure",
        0.45,
        asof_utc=asof_utc,
        path="/api/darkpool/{ticker}",
        use_role="report-only",
        decision_eligible=False,
    )

    pred_without = _run(base_features, cfg)
    pred_with = _run(base_features + [report_only_darkpool], cfg)

    assert pred_without["decision_state"] == pred_with["decision_state"]
    assert pred_without["data_quality_state"] == pred_with["data_quality_state"]
    assert pred_without["confidence"] == pred_with["confidence"]
    assert pred_without["meta_json"]["decision_dq"] == pred_with["meta_json"]["decision_dq"] == 1.0
    assert pred_without["meta_json"]["dq_reason_codes"] == pred_with["meta_json"]["dq_reason_codes"] == []
    assert pred_with["meta_json"]["horizon_contract"]["report_only_excluded_features"] == ["darkpool_pressure"]



def test_zero_weight_feature_absence_does_not_degrade_confidence():
    asof_utc = dt.datetime(2026, 3, 3, 15, 0, tzinfo=dt.timezone.utc)
    cfg = _cfg(weights={"oi_pressure": 1.0, "darkpool_pressure": 0.0}, criticals=["spot"])
    pred = _run(
        [
            _feature("spot", 150.0, asof_utc=asof_utc, path="/api/stock/{ticker}/ohlc/{candle_size}"),
            _feature("oi_pressure", 0.6, asof_utc=asof_utc, path="/api/stock/{ticker}/oi-per-strike"),
        ],
        cfg,
    )

    assert pred["data_quality_state"] == "VALID"
    assert pred["meta_json"]["decision_dq"] == 1.0
    assert not any(reason.startswith("darkpool_pressure") for reason in pred["meta_json"]["dq_reason_codes"])
    assert pred["meta_json"]["horizon_contract"]["zero_weight_excluded_features"] == ["darkpool_pressure"]



def test_context_only_feature_is_excluded_and_logs_contract_violation(caplog):
    asof_utc = dt.datetime(2026, 3, 3, 15, 0, tzinfo=dt.timezone.utc)
    cfg = _cfg(weights={"oi_pressure": 1.0, "vol_term_slope": 1.0}, criticals=["spot"])
    features = [
        _feature("spot", 150.0, asof_utc=asof_utc, path="/api/stock/{ticker}/ohlc/{candle_size}"),
        _feature("oi_pressure", 0.55, asof_utc=asof_utc, path="/api/stock/{ticker}/oi-per-strike"),
        _feature(
            "vol_term_slope",
            None,
            asof_utc=asof_utc,
            path="/api/market/market-tide",
            use_role="context-only",
            decision_eligible=False,
        ),
    ]

    with caplog.at_level(logging.INFO):
        pred = _run(features, cfg)

    assert pred["data_quality_state"] == "VALID"
    assert pred["meta_json"]["decision_dq"] == 1.0
    assert pred["meta_json"]["dq_reason_codes"] == []
    assert pred["meta_json"]["horizon_contract"]["context_only_excluded_features"] == ["vol_term_slope"]
    assert pred["meta_json"]["horizon_contract"]["contract_violation_features"] == ["vol_term_slope"]

    counters = [getattr(record, "counter", None) for record in caplog.records]
    assert "context_only_feature_excluded_count" in counters
    assert "decision_path_contract_violation_count" in counters
