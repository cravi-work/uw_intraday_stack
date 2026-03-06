import datetime as dt

from src.freshness_policy import assess_feature_freshness, resolve_feature_policy


def _runtime_cfg(alignment_tolerance_sec: int = 900, fallback_max_age_minutes: int = 15) -> dict:
    return {
        "validation": {
            "alignment_tolerance_sec": alignment_tolerance_sec,
            "fallback_max_age_minutes": fallback_max_age_minutes,
        }
    }


def test_options_snapshot_policy_has_week_join_skew_guardrail():
    cfg = _runtime_cfg()
    meta_json = {
        "metric_lineage": {
            "effective_ts_utc": "2026-03-06T00:00:00Z",
        },
        "source_endpoints": [{"method": "GET", "path": "/api/stock/{ticker}/oi-per-strike"}],
        "freshness_state": "FRESH",
        "time_provenance": "SOURCE_EVENT_TIME",
    }

    policy = resolve_feature_policy("oi_pressure", meta_json, cfg)

    assert policy.join_skew_tolerance_seconds == 7 * 24 * 3600
    assert policy.max_tolerated_age_seconds == 7 * 24 * 3600


def test_snapshot_feature_15h_old_is_not_rejected_by_join_skew_policy():
    cfg = _runtime_cfg()

    # 2026-03-06 15:16:35Z (roughly 10:16am ET).
    asof = dt.datetime(2026, 3, 6, 15, 16, 35, tzinfo=dt.timezone.utc)
    # Snapshot timestamp ~15h 15m earlier.
    eff = asof - dt.timedelta(seconds=55_000)

    feature = {
        "feature_key": "oi_pressure",
        "feature_value": 0.12,
        "meta_json": {
            "metric_lineage": {
                "effective_ts_utc": eff.isoformat().replace("+00:00", "Z"),
            },
            "source_endpoints": [{"method": "GET", "path": "/api/stock/{ticker}/oi-per-strike"}],
            "freshness_state": "FRESH",
            "stale_age_min": 0,
            "time_provenance": "SOURCE_EVENT_TIME",
        },
    }

    assessment = assess_feature_freshness(feature, asof_utc=asof, cadence_seconds=60, cfg=cfg)

    assert assessment.include_in_scoring is True
    assert assessment.reason.value == "ok"
