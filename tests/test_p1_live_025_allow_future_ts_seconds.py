import datetime as dt


from src.freshness_policy import assess_feature_freshness, FeatureDecisionReason


ASOF_UTC = dt.datetime(2026, 3, 6, 20, 50, tzinfo=dt.timezone.utc)


def _feature(feature_key: str, feature_value: float, effective_ts: dt.datetime) -> dict:
    return {
        "feature_key": feature_key,
        "feature_value": feature_value,
        "meta_json": {
            "freshness_state": "FRESH",
            "stale_age_min": 0,
            "metric_lineage": {
                "effective_ts_utc": effective_ts.isoformat(),
                "time_provenance_degraded": False,
            }
        },
    }


def test_future_timestamp_clamped_when_allow_future_ts_seconds_is_wider_than_cadence() -> None:
    cfg = {
        "validation": {
            "allow_future_ts_seconds": 600,
            "alignment_tolerance_sec": 60,
            "invalid_after_minutes": 90,
        }
    }
    # Provider timestamp is ~6 minutes ahead of as-of.
    eff_ts = ASOF_UTC + dt.timedelta(minutes=6)
    feat = _feature("spot", 123.45, eff_ts)

    assessment = assess_feature_freshness(feat, asof_utc=ASOF_UTC, cadence_seconds=60, cfg=cfg)

    assert assessment.reason == FeatureDecisionReason.OK
    assert assessment.include_in_scoring is True
    assert assessment.normalized_future_ts is True
    assert assessment.future_drift_seconds == 360
    assert assessment.degraded is True
    assert assessment.dq_reason_code == "spot_future_ts_clamped_360s"
    assert assessment.effective_ts == ASOF_UTC
    assert assessment.delta_seconds == 0


def test_future_timestamp_rejected_when_allow_future_ts_seconds_not_set() -> None:
    cfg = {
        "validation": {
            "alignment_tolerance_sec": 60,
            "invalid_after_minutes": 90,
        }
    }
    eff_ts = ASOF_UTC + dt.timedelta(minutes=6)
    feat = _feature("spot", 123.45, eff_ts)

    assessment = assess_feature_freshness(feat, asof_utc=ASOF_UTC, cadence_seconds=60, cfg=cfg)

    assert assessment.reason == FeatureDecisionReason.FUTURE_TS_VIOLATION
    assert assessment.include_in_scoring is False
    assert assessment.normalized_future_ts is False
    assert assessment.future_drift_seconds == 360
