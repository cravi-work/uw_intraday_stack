from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


class EndpointCriticality(str, Enum):
    CRITICAL = "CRITICAL"
    NON_CRITICAL = "NON_CRITICAL"


class LagClass(str, Enum):
    LIVE = "LIVE"
    NEAR_REALTIME = "NEAR_REALTIME"
    SNAPSHOT = "SNAPSHOT"
    CONTEXT = "CONTEXT"
    UNKNOWN = "UNKNOWN"


class PolicyAction(str, Enum):
    ACCEPT = "ACCEPT"
    DEGRADE = "DEGRADE"
    SUPPRESS = "SUPPRESS"


class FeatureDecisionReason(str, Enum):
    OK = "ok"
    MISSING_EFFECTIVE_TS = "missing_effective_ts"
    INVALID_EFFECTIVE_TS = "invalid_effective_ts"
    FUTURE_TS_VIOLATION = "future_ts_violation"
    JOIN_SKEW_VIOLATION = "join_skew_violation"
    BAD_FRESHNESS = "bad_freshness"
    STALE_ENDPOINT_REJECTED = "stale_endpoint_rejected"
    CARRY_FORWARD_SUPPRESSED = "carry_forward_suppressed"
    TIME_PROVENANCE_SUPPRESSED = "time_provenance_suppressed"
    MISSING_OR_INVALID_VALUE = "missing_or_invalid_value"


@dataclass(frozen=True)
class EndpointFreshnessPolicy:
    name: str
    method: str = "GET"
    path_patterns: Tuple[str, ...] = ()
    max_tolerated_age_seconds: int = 5400
    join_skew_tolerance_seconds: int = 900
    criticality: EndpointCriticality = EndpointCriticality.NON_CRITICAL
    lag_class: LagClass = LagClass.SNAPSHOT
    fresh_behavior: PolicyAction = PolicyAction.ACCEPT
    stale_behavior: PolicyAction = PolicyAction.DEGRADE
    carry_forward_behavior: PolicyAction = PolicyAction.DEGRADE
    empty_valid_behavior: PolicyAction = PolicyAction.ACCEPT
    time_provenance_degraded_behavior: PolicyAction = PolicyAction.DEGRADE


@dataclass(frozen=True)
class ResolvedFeaturePolicy:
    name: str
    max_tolerated_age_seconds: int
    join_skew_tolerance_seconds: int
    criticality: EndpointCriticality
    lag_class: LagClass
    fresh_behavior: PolicyAction
    stale_behavior: PolicyAction
    carry_forward_behavior: PolicyAction
    empty_valid_behavior: PolicyAction
    time_provenance_degraded_behavior: PolicyAction
    sources: Tuple[str, ...]


@dataclass(frozen=True)
class FeaturePolicyAssessment:
    feature_key: str
    policy: ResolvedFeaturePolicy
    include_in_scoring: bool
    degraded: bool
    reason: FeatureDecisionReason
    reason_detail: Optional[str]
    dq_reason_code: Optional[str]
    effective_ts: Optional[dt.datetime]
    delta_seconds: Optional[int]
    normalized_future_ts: bool
    freshness_state: str
    stale_age_seconds: Optional[int]
    stale_age_minutes: Optional[int]
    time_provenance_degraded: bool
    policy_source: str
    future_drift_seconds: Optional[int] = None


def _cfg_scope(cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(cfg, dict) and isinstance(cfg.get("validation"), dict):
        return cfg["validation"]
    return cfg


def _default_max_age_seconds(cfg: Mapping[str, Any]) -> int:
    scoped = _cfg_scope(cfg)
    invalid_after_minutes = scoped.get("invalid_after_minutes")
    if isinstance(invalid_after_minutes, int) and invalid_after_minutes > 0:
        return int(invalid_after_minutes) * 60
    return 5400


def _default_join_skew_seconds(cfg: Mapping[str, Any]) -> int:
    scoped = _cfg_scope(cfg)
    tol = scoped.get("alignment_tolerance_sec")
    if isinstance(tol, int) and tol > 0:
        return int(tol)
    return 900


def _clamp_positive_int(value: Any, fallback: int) -> int:
    try:
        ivalue = int(value)
    except (TypeError, ValueError):
        return fallback
    return ivalue if ivalue > 0 else fallback


def _merge_action(actions: Iterable[PolicyAction], fallback: PolicyAction) -> PolicyAction:
    ordered = list(actions)
    if not ordered:
        return fallback
    if PolicyAction.SUPPRESS in ordered:
        return PolicyAction.SUPPRESS
    if PolicyAction.DEGRADE in ordered:
        return PolicyAction.DEGRADE
    return PolicyAction.ACCEPT


def _merge_lag_class(values: Iterable[LagClass]) -> LagClass:
    order = {
        LagClass.LIVE: 0,
        LagClass.NEAR_REALTIME: 1,
        LagClass.SNAPSHOT: 2,
        LagClass.CONTEXT: 3,
        LagClass.UNKNOWN: 4,
    }
    selected = min(list(values) or [LagClass.UNKNOWN], key=lambda v: order.get(v, 999))
    return selected


def _matches_path(pattern: str, path: str) -> bool:
    if pattern == path:
        return True
    if pattern.endswith("*"):
        return path.startswith(pattern[:-1])
    return pattern in path


_ENDPOINT_POLICIES: Tuple[EndpointFreshnessPolicy, ...] = (
    EndpointFreshnessPolicy(
        name="live_ohlc_price",
        path_patterns=("/api/stock/{ticker}/ohlc/", "/api/stock/{ticker}/ohlc/{candle_size}"),
        max_tolerated_age_seconds=60,
        join_skew_tolerance_seconds=60,
        criticality=EndpointCriticality.CRITICAL,
        lag_class=LagClass.LIVE,
        stale_behavior=PolicyAction.SUPPRESS,
        carry_forward_behavior=PolicyAction.SUPPRESS,
        time_provenance_degraded_behavior=PolicyAction.SUPPRESS,
    ),
    EndpointFreshnessPolicy(
        name="intraday_option_flow",
        path_patterns=(
            "/api/stock/{ticker}/flow-recent",
            "/api/stock/{ticker}/flow-per-strike-intraday",
            "/api/stock/{ticker}/flow-per-strike",
            "/api/lit-flow/{ticker}",
            "/api/darkpool/{ticker}",
        ),
        max_tolerated_age_seconds=300,
        join_skew_tolerance_seconds=300,
        criticality=EndpointCriticality.CRITICAL,
        lag_class=LagClass.NEAR_REALTIME,
        stale_behavior=PolicyAction.SUPPRESS,
        carry_forward_behavior=PolicyAction.SUPPRESS,
        time_provenance_degraded_behavior=PolicyAction.DEGRADE,
    ),
    EndpointFreshnessPolicy(
        name="options_snapshot",
        path_patterns=(
            "/api/stock/{ticker}/oi-per-strike",
            "/api/stock/{ticker}/oi-change",
            "/api/stock/{ticker}/spot-exposures",
            "/api/stock/{ticker}/spot-exposures/strike",
            "/api/stock/{ticker}/spot-exposures/expiry-strike",
            "/api/stock/{ticker}/greek-exposure",
            "/api/stock/{ticker}/greek-exposure/strike",
            "/api/stock/{ticker}/greek-exposure/expiry",
            "/api/stock/{ticker}/iv-rank",
            "/api/stock/{ticker}/volatility/term-structure",
            "/api/stock/{ticker}/historical-risk-reversal-skew",
        ),
        max_tolerated_age_seconds=7 * 24 * 3600,
        join_skew_tolerance_seconds=7 * 24 * 3600,
        criticality=EndpointCriticality.NON_CRITICAL,
        lag_class=LagClass.SNAPSHOT,
        stale_behavior=PolicyAction.DEGRADE,
        carry_forward_behavior=PolicyAction.DEGRADE,
        time_provenance_degraded_behavior=PolicyAction.DEGRADE,
    ),
    EndpointFreshnessPolicy(
        name="market_context",
        path_patterns=(
            "/api/market/market-tide",
            "/api/market/economic-calendar",
            "/api/market/top-net-impact",
            "/api/market/total-options-volume",
            "/api/market",
        ),
        max_tolerated_age_seconds=5400,
        join_skew_tolerance_seconds=1800,
        criticality=EndpointCriticality.NON_CRITICAL,
        lag_class=LagClass.CONTEXT,
        stale_behavior=PolicyAction.DEGRADE,
        carry_forward_behavior=PolicyAction.DEGRADE,
        time_provenance_degraded_behavior=PolicyAction.DEGRADE,
    ),
)


_FEATURE_KEY_FALLBACK_POLICIES: Mapping[str, str] = {
    "spot": "generic_default",
    "smart_whale_pressure": "generic_default",
    "dealer_vanna": "generic_default",
    "dealer_charm": "generic_default",
    "net_gex_sign": "generic_default",
    "net_gamma_exposure_notional": "generic_default",
    "oi_pressure": "generic_default",
    "darkpool_pressure": "generic_default",
    "litflow_pressure": "generic_default",
    "vol_term_slope": "generic_default",
    "vol_skew": "generic_default",
    "iv_rank": "generic_default",
}


def default_endpoint_policy(cfg: Mapping[str, Any]) -> EndpointFreshnessPolicy:
    max_age = _default_max_age_seconds(cfg)
    join_skew = _default_join_skew_seconds(cfg)
    return EndpointFreshnessPolicy(
        name="generic_default",
        max_tolerated_age_seconds=max_age,
        join_skew_tolerance_seconds=join_skew,
        criticality=EndpointCriticality.NON_CRITICAL,
        lag_class=LagClass.UNKNOWN,
        stale_behavior=PolicyAction.DEGRADE,
        carry_forward_behavior=PolicyAction.DEGRADE,
        time_provenance_degraded_behavior=PolicyAction.DEGRADE,
    )


def resolve_endpoint_policy(method: str, path: str, cfg: Mapping[str, Any]) -> EndpointFreshnessPolicy:
    defaults = default_endpoint_policy(cfg)
    for policy in _ENDPOINT_POLICIES:
        if method and policy.method and method.upper() != policy.method.upper():
            continue
        if any(_matches_path(pattern, path) for pattern in policy.path_patterns):
            return EndpointFreshnessPolicy(
                name=policy.name,
                method=policy.method,
                path_patterns=policy.path_patterns,
                max_tolerated_age_seconds=policy.max_tolerated_age_seconds,
                join_skew_tolerance_seconds=policy.join_skew_tolerance_seconds,
                criticality=policy.criticality,
                lag_class=policy.lag_class,
                fresh_behavior=policy.fresh_behavior,
                stale_behavior=policy.stale_behavior,
                carry_forward_behavior=policy.carry_forward_behavior,
                empty_valid_behavior=policy.empty_valid_behavior,
                time_provenance_degraded_behavior=policy.time_provenance_degraded_behavior,
            )
    return defaults


def resolve_feature_policy(feature_key: str, meta_json: Mapping[str, Any], cfg: Mapping[str, Any]) -> ResolvedFeaturePolicy:
    source_endpoints = meta_json.get("source_endpoints", [])
    endpoint_policies: List[EndpointFreshnessPolicy] = []

    for ep in source_endpoints:
        if not isinstance(ep, Mapping):
            continue
        method = str(ep.get("method") or "GET")
        path = str(ep.get("path") or "")
        endpoint_policies.append(resolve_endpoint_policy(method, path, cfg))

    if not endpoint_policies:
        fallback_name = _FEATURE_KEY_FALLBACK_POLICIES.get(feature_key, "generic_default")
        if fallback_name == "generic_default":
            endpoint_policies = [default_endpoint_policy(cfg)]
        else:
            endpoint_policies = [default_endpoint_policy(cfg)]

    return ResolvedFeaturePolicy(
        name="|".join(sorted({p.name for p in endpoint_policies})),
        max_tolerated_age_seconds=max(p.max_tolerated_age_seconds for p in endpoint_policies),
        join_skew_tolerance_seconds=max(p.join_skew_tolerance_seconds for p in endpoint_policies),
        criticality=EndpointCriticality.CRITICAL if any(p.criticality == EndpointCriticality.CRITICAL for p in endpoint_policies) else EndpointCriticality.NON_CRITICAL,
        lag_class=_merge_lag_class(p.lag_class for p in endpoint_policies),
        fresh_behavior=_merge_action((p.fresh_behavior for p in endpoint_policies), PolicyAction.ACCEPT),
        stale_behavior=_merge_action((p.stale_behavior for p in endpoint_policies), PolicyAction.DEGRADE),
        carry_forward_behavior=_merge_action((p.carry_forward_behavior for p in endpoint_policies), PolicyAction.DEGRADE),
        empty_valid_behavior=_merge_action((p.empty_valid_behavior for p in endpoint_policies), PolicyAction.ACCEPT),
        time_provenance_degraded_behavior=_merge_action(
            (p.time_provenance_degraded_behavior for p in endpoint_policies),
            PolicyAction.DEGRADE,
        ),
        sources=tuple(sorted({p.name for p in endpoint_policies})),
    )


def _parse_effective_ts(raw: Any) -> Tuple[Optional[dt.datetime], Optional[FeatureDecisionReason], Optional[str]]:
    if raw is None:
        return None, FeatureDecisionReason.MISSING_EFFECTIVE_TS, "missing_effective_ts"
    if isinstance(raw, str):
        if not raw.strip():
            return None, FeatureDecisionReason.MISSING_EFFECTIVE_TS, "missing_effective_ts"
        if raw == "INVALID":
            return None, FeatureDecisionReason.INVALID_EFFECTIVE_TS, "invalid_effective_ts"
        try:
            parsed = dt.datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return None, FeatureDecisionReason.INVALID_EFFECTIVE_TS, "invalid_effective_ts"
    elif isinstance(raw, dt.datetime):
        parsed = raw
    else:
        return None, FeatureDecisionReason.INVALID_EFFECTIVE_TS, "invalid_effective_ts"

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    else:
        parsed = parsed.astimezone(dt.timezone.utc)
    return parsed, None, None


def _value_is_valid(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def _carry_forward_max_age_seconds(policy: ResolvedFeaturePolicy, meta_json: Mapping[str, Any]) -> int:
    """Allow relaxed join-skew for snapshot endpoints while keeping stale carry stricter for OI."""
    endpoints = meta_json.get("source_endpoints") if isinstance(meta_json, Mapping) else None
    if isinstance(endpoints, list):
        paths = {str(ep.get("path") or "") for ep in endpoints if isinstance(ep, Mapping)}
        if "/api/stock/{ticker}/oi-per-strike" in paths or "/api/stock/{ticker}/oi-change" in paths:
            return 5400
    return policy.max_tolerated_age_seconds


def assess_feature_freshness(
    feature: Mapping[str, Any],
    *,
    asof_utc: dt.datetime,
    cadence_seconds: int,
    cfg: Mapping[str, Any],
) -> FeaturePolicyAssessment:
    feature_key = str(feature.get("feature_key") or "")
    meta_json = feature.get("meta_json", {}) or {}
    policy = resolve_feature_policy(feature_key, meta_json, cfg)

    metric_lineage = meta_json.get("metric_lineage", {}) or {}
    feature_value = feature.get("feature_value")
    freshness_state = str(meta_json.get("freshness_state") or "ERROR")
    stale_age_minutes = meta_json.get("stale_age_min")
    stale_age_seconds = None if stale_age_minutes is None else _clamp_positive_int(stale_age_minutes, 0) * 60
    time_provenance_degraded = bool(
        metric_lineage.get("time_provenance_degraded")
        or meta_json.get("details", {}).get("time_provenance_degraded")
    )

    eff_ts, ts_error, ts_error_detail = _parse_effective_ts(metric_lineage.get("effective_ts_utc"))
    if ts_error is not None:
        return FeaturePolicyAssessment(
            feature_key=feature_key,
            policy=policy,
            include_in_scoring=False,
            degraded=False,
            reason=ts_error,
            reason_detail=ts_error_detail or ts_error.value,
            dq_reason_code=f"{feature_key}_{ts_error.value}",
            effective_ts=None,
            delta_seconds=None,
            normalized_future_ts=False,
            freshness_state=freshness_state,
            stale_age_seconds=stale_age_seconds,
            stale_age_minutes=stale_age_minutes if isinstance(stale_age_minutes, int) else None,
            time_provenance_degraded=time_provenance_degraded,
            policy_source=",".join(policy.sources) if policy.sources else "default",
        )

    # STRICT DEFAULT:
    # if allow_future_ts_seconds is not explicitly configured, future timestamps are rejected.
    allow_future_ts_seconds = 0
    validation_cfg = cfg.get("validation", {}) if isinstance(cfg, dict) else {}
    if isinstance(validation_cfg, Mapping):
        raw_allow = validation_cfg.get("allow_future_ts_seconds", None)
        if raw_allow is not None:
            try:
                allow_future_ts_seconds = max(int(raw_allow), 0)
            except Exception:
                allow_future_ts_seconds = 0

    normalized_future_ts = False
    future_drift_seconds: Optional[int] = None
    delta_seconds = int((asof_utc - eff_ts).total_seconds())

    if delta_seconds < 0:
        drift = abs(delta_seconds)
        if allow_future_ts_seconds > 0 and drift <= allow_future_ts_seconds:
            future_drift_seconds = drift
            eff_ts = asof_utc
            delta_seconds = 0
            normalized_future_ts = True
            time_provenance_degraded = True
        else:
            return FeaturePolicyAssessment(
                feature_key=feature_key,
                policy=policy,
                include_in_scoring=False,
                degraded=False,
                reason=FeatureDecisionReason.FUTURE_TS_VIOLATION,
                reason_detail=f"ahead_by_{drift}s_exceeds_allow_future={allow_future_ts_seconds}s",
                dq_reason_code=f"{feature_key}_{FeatureDecisionReason.FUTURE_TS_VIOLATION.value}",
                effective_ts=eff_ts,
                delta_seconds=-drift,
                normalized_future_ts=False,
                freshness_state=freshness_state,
                stale_age_seconds=stale_age_seconds,
                stale_age_minutes=stale_age_minutes if isinstance(stale_age_minutes, int) else None,
                time_provenance_degraded=time_provenance_degraded,
                policy_source=",".join(policy.sources) if policy.sources else "default",
                future_drift_seconds=drift,
            )

    def _carry_forward_max_age_seconds() -> int:
        endpoints = meta_json.get("source_endpoints") if isinstance(meta_json, Mapping) else None
        if isinstance(endpoints, list):
            paths = {str(ep.get("path") or "") for ep in endpoints if isinstance(ep, Mapping)}
            if "/api/stock/{ticker}/oi-per-strike" in paths or "/api/stock/{ticker}/oi-change" in paths:
                return 5400
        return policy.max_tolerated_age_seconds

    carry_forward_max_age_seconds = _carry_forward_max_age_seconds()

    if freshness_state == "STALE_CARRY" and stale_age_seconds is not None and stale_age_seconds > carry_forward_max_age_seconds:
        return FeaturePolicyAssessment(
            feature_key=feature_key,
            policy=policy,
            include_in_scoring=False,
            degraded=False,
            reason=FeatureDecisionReason.STALE_ENDPOINT_REJECTED,
            reason_detail=f"age_{stale_age_seconds}s_gt_{carry_forward_max_age_seconds}s",
            dq_reason_code=f"{feature_key}_{FeatureDecisionReason.STALE_ENDPOINT_REJECTED.value}",
            effective_ts=eff_ts,
            delta_seconds=delta_seconds,
            normalized_future_ts=normalized_future_ts,
            freshness_state=freshness_state,
            stale_age_seconds=stale_age_seconds,
            stale_age_minutes=stale_age_minutes if isinstance(stale_age_minutes, int) else None,
            time_provenance_degraded=time_provenance_degraded,
            policy_source=",".join(policy.sources) if policy.sources else "default",
            future_drift_seconds=future_drift_seconds,
        )

    if delta_seconds > policy.join_skew_tolerance_seconds:
        return FeaturePolicyAssessment(
            feature_key=feature_key,
            policy=policy,
            include_in_scoring=False,
            degraded=False,
            reason=FeatureDecisionReason.JOIN_SKEW_VIOLATION,
            reason_detail=f"delta_{delta_seconds}s_gt_{policy.join_skew_tolerance_seconds}s",
            dq_reason_code=f"{feature_key}_{FeatureDecisionReason.JOIN_SKEW_VIOLATION.value}",
            effective_ts=eff_ts,
            delta_seconds=delta_seconds,
            normalized_future_ts=normalized_future_ts,
            freshness_state=freshness_state,
            stale_age_seconds=stale_age_seconds,
            stale_age_minutes=stale_age_minutes if isinstance(stale_age_minutes, int) else None,
            time_provenance_degraded=time_provenance_degraded,
            policy_source=",".join(policy.sources) if policy.sources else "default",
            future_drift_seconds=future_drift_seconds,
        )

    if not _value_is_valid(feature_value):
        return FeaturePolicyAssessment(
            feature_key=feature_key,
            policy=policy,
            include_in_scoring=False,
            degraded=False,
            reason=FeatureDecisionReason.MISSING_OR_INVALID_VALUE,
            reason_detail=None,
            dq_reason_code=f"{feature_key}_{FeatureDecisionReason.MISSING_OR_INVALID_VALUE.value}",
            effective_ts=eff_ts,
            delta_seconds=delta_seconds,
            normalized_future_ts=normalized_future_ts,
            freshness_state=freshness_state,
            stale_age_seconds=stale_age_seconds,
            stale_age_minutes=stale_age_minutes if isinstance(stale_age_minutes, int) else None,
            time_provenance_degraded=time_provenance_degraded,
            policy_source=",".join(policy.sources) if policy.sources else "default",
        )

    if time_provenance_degraded and policy.time_provenance_degraded_behavior == PolicyAction.SUPPRESS:
        return FeaturePolicyAssessment(
            feature_key=feature_key,
            policy=policy,
            include_in_scoring=False,
            degraded=False,
            reason=FeatureDecisionReason.TIME_PROVENANCE_SUPPRESSED,
            reason_detail="time_provenance_degraded",
            dq_reason_code=f"{feature_key}_{FeatureDecisionReason.TIME_PROVENANCE_SUPPRESSED.value}",
            effective_ts=eff_ts,
            delta_seconds=delta_seconds,
            normalized_future_ts=normalized_future_ts,
            freshness_state=freshness_state,
            stale_age_seconds=stale_age_seconds,
            stale_age_minutes=stale_age_minutes if isinstance(stale_age_minutes, int) else None,
            time_provenance_degraded=True,
            policy_source=",".join(policy.sources) if policy.sources else "default",
            future_drift_seconds=future_drift_seconds,
        )

    degraded = False
    dq_reason = None

    if freshness_state in ("FRESH",):
        action = policy.fresh_behavior
    elif freshness_state == "EMPTY_VALID":
        action = policy.empty_valid_behavior
    elif freshness_state == "STALE_CARRY":
        age = stale_age_seconds
        if age is not None and age > carry_forward_max_age_seconds:
            return FeaturePolicyAssessment(
                feature_key=feature_key,
                policy=policy,
                include_in_scoring=False,
                degraded=False,
                reason=FeatureDecisionReason.STALE_ENDPOINT_REJECTED,
                reason_detail=f"age_{age}s_gt_{carry_forward_max_age_seconds}s",
                dq_reason_code=f"{feature_key}_{FeatureDecisionReason.STALE_ENDPOINT_REJECTED.value}",
                effective_ts=eff_ts,
                delta_seconds=delta_seconds,
                normalized_future_ts=normalized_future_ts,
                freshness_state=freshness_state,
                stale_age_seconds=age,
                stale_age_minutes=stale_age_minutes if isinstance(stale_age_minutes, int) else None,
                time_provenance_degraded=time_provenance_degraded,
                policy_source=",".join(policy.sources) if policy.sources else "default",
                future_drift_seconds=future_drift_seconds,
            )
        if policy.carry_forward_behavior == PolicyAction.SUPPRESS:
            return FeaturePolicyAssessment(
                feature_key=feature_key,
                policy=policy,
                include_in_scoring=False,
                degraded=False,
                reason=FeatureDecisionReason.CARRY_FORWARD_SUPPRESSED,
                reason_detail=f"lag_class={policy.lag_class.value}",
                dq_reason_code=f"{feature_key}_{FeatureDecisionReason.CARRY_FORWARD_SUPPRESSED.value}",
                effective_ts=eff_ts,
                delta_seconds=delta_seconds,
                normalized_future_ts=normalized_future_ts,
                freshness_state=freshness_state,
                stale_age_seconds=age,
                stale_age_minutes=stale_age_minutes if isinstance(stale_age_minutes, int) else None,
                time_provenance_degraded=time_provenance_degraded,
                policy_source=",".join(policy.sources) if policy.sources else "default",
                future_drift_seconds=future_drift_seconds,
            )
        action = policy.carry_forward_behavior
        degraded = action == PolicyAction.DEGRADE
        if stale_age_minutes is not None:
            dq_reason = f"{feature_key}_stale_carry_age_{int(stale_age_minutes)}m"
        else:
            dq_reason = f"{feature_key}_stale_carry"
    else:
        return FeaturePolicyAssessment(
            feature_key=feature_key,
            policy=policy,
            include_in_scoring=False,
            degraded=False,
            reason=FeatureDecisionReason.BAD_FRESHNESS,
            reason_detail=freshness_state,
            dq_reason_code=f"{feature_key}_{FeatureDecisionReason.BAD_FRESHNESS.value}_{freshness_state}",
            effective_ts=eff_ts,
            delta_seconds=delta_seconds,
            normalized_future_ts=normalized_future_ts,
            freshness_state=freshness_state,
            stale_age_seconds=stale_age_seconds,
            stale_age_minutes=stale_age_minutes if isinstance(stale_age_minutes, int) else None,
            time_provenance_degraded=time_provenance_degraded,
            policy_source=",".join(policy.sources) if policy.sources else "default",
            future_drift_seconds=future_drift_seconds,
        )

    if action == PolicyAction.SUPPRESS:
        return FeaturePolicyAssessment(
            feature_key=feature_key,
            policy=policy,
            include_in_scoring=False,
            degraded=False,
            reason=FeatureDecisionReason.STALE_ENDPOINT_REJECTED,
            reason_detail=f"policy={policy.name}",
            dq_reason_code=f"{feature_key}_{FeatureDecisionReason.STALE_ENDPOINT_REJECTED.value}",
            effective_ts=eff_ts,
            delta_seconds=delta_seconds,
            normalized_future_ts=normalized_future_ts,
            freshness_state=freshness_state,
            stale_age_seconds=stale_age_seconds,
            stale_age_minutes=stale_age_minutes if isinstance(stale_age_minutes, int) else None,
            time_provenance_degraded=time_provenance_degraded,
            policy_source=",".join(policy.sources) if policy.sources else "default",
            future_drift_seconds=future_drift_seconds,
        )

    if future_drift_seconds is not None:
        degraded = True
        dq_reason = f"{feature_key}_future_ts_clamped_{future_drift_seconds}s"

    if time_provenance_degraded:
        degraded = True
        if "_future_ts_clamped_" not in (dq_reason or ""):
            dq_reason = dq_reason or f"{feature_key}_time_provenance_degraded"

    return FeaturePolicyAssessment(
        feature_key=feature_key,
        policy=policy,
        include_in_scoring=True,
        degraded=degraded,
        reason=FeatureDecisionReason.OK,
        reason_detail=None,
        dq_reason_code=dq_reason,
        effective_ts=eff_ts,
        delta_seconds=delta_seconds,
        normalized_future_ts=normalized_future_ts,
        freshness_state=freshness_state,
        stale_age_seconds=stale_age_seconds,
        stale_age_minutes=stale_age_minutes if isinstance(stale_age_minutes, int) else None,
        time_provenance_degraded=time_provenance_degraded,
        policy_source=",".join(policy.sources) if policy.sources else "default",
        future_drift_seconds=future_drift_seconds,
    )