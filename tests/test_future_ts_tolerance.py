from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.freshness_policy import assess_feature_freshness


def _mk_feature(feature_key: str, effective_ts_utc: datetime) -> dict:
    return {
        "feature_key": feature_key,
        "feature_value": 1.0,
        "meta_json": {
            # In the real pipeline, feature freshness_state is computed upstream.
            # For unit tests, we set a valid state so the policy engine evaluates
            # only the join-skew / future-ts logic.
            "freshness_state": "FRESH",
            "metric_lineage": {
                # The pipeline stores timestamps as ISO strings.
                "effective_ts_utc": effective_ts_utc.isoformat(),
            }
        },
    }


def test_future_ts_default_tolerance_forward_observation_normalizes_small_skew() -> None:
    """In forward observation, we tolerate a few minutes of host clock skew by default."""

    asof = datetime(2026, 3, 6, 20, 35, 0, tzinfo=timezone.utc)
    eff = asof + timedelta(seconds=360)

    cfg = {
        "ingest": {"cadence_minutes": 5},
        "validation": {"governance_mode": "FORWARD_OBSERVATION"},
    }

    assessment = assess_feature_freshness(
        feature=_mk_feature("spot", eff),
        asof_utc=asof,
        cadence_seconds=300,
        cfg=cfg,
    )

    assert assessment.include_in_scoring is True
    assert assessment.normalized_future_ts is True
    assert assessment.delta_seconds == 0


def test_future_ts_explicit_tolerance_is_enforced() -> None:
    asof = datetime(2026, 3, 6, 20, 35, 0, tzinfo=timezone.utc)
    eff = asof + timedelta(seconds=360)

    cfg = {
        "ingest": {"cadence_minutes": 5},
        "validation": {
            "governance_mode": "FORWARD_OBSERVATION",
            "allow_future_ts_seconds": 300,
        },
    }

    assessment = assess_feature_freshness(
        feature=_mk_feature("spot", eff),
        asof_utc=asof,
        cadence_seconds=300,
        cfg=cfg,
    )

    assert assessment.include_in_scoring is False
    assert assessment.normalized_future_ts is False
    assert assessment.reason.value == "future_ts_violation"
    assert "exceeds_allow_future=300s" in (assessment.reason_detail or "")


def test_future_ts_institutional_grade_does_not_widen_default_window() -> None:
    """Institutional-grade should remain strict-by-default when allow_future is not configured."""

    asof = datetime(2026, 3, 6, 20, 35, 0, tzinfo=timezone.utc)
    eff = asof + timedelta(seconds=360)

    cfg = {
        "ingest": {"cadence_minutes": 5},
        "validation": {"governance_mode": "INSTITUTIONAL_GRADE"},
    }

    assessment = assess_feature_freshness(
        feature=_mk_feature("spot", eff),
        asof_utc=asof,
        cadence_seconds=300,
        cfg=cfg,
    )

    assert assessment.include_in_scoring is False
    assert assessment.reason.value == "future_ts_violation"
