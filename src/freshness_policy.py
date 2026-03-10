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
    # When we clamp an effective timestamp that is slightly ahead of asof_utc, record
    # the original observed forward drift (seconds). This helps diagnose local clock skew.
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
        # Unusual Whales "snapshot" endpoints (OI, vol surfaces, exposures) often update
        # at session cadence rather than true intraday cadence. During LIVE runs this means a
        # valid snapshot can be ~15-72h old (prior session / weekend) even though the payload
        # itself is "fresh" (200 OK) and structurally valid.
        #
        # If we keep a tight join-skew (e.g., 1.5h), the engine will drop these inputs and can
        # end up with near-zero feature coverage, triggering OOD + risk-gate blocks.
        #
        # Guardrail: we still reject extremely old snapshots (weeks/months) by bounding the
        # allowable join-skew/age window.
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
                max_tolerated_age_seconds=_clamp_positive_int(policy.max_tolerated_age_seconds, defaults.max_tolerated_age_seconds),
                join_skew_tolerance_seconds=_clamp_positive_int(policy.join_skew_tolerance_seconds, defaults.join_skew_tolerance_seconds),
                criticality=policy.criticality,
                lag_class=policy.lag_class,
                fresh_behavior=policy.fresh_behavior,
                stale_behavior=policy.stale_behavior,
                carry_forward_behavior=policy.carry_forward_behavior,
                empty_valid_behavior=policy.empty_valid_behavior,
                time_provenance_degraded_behavior=policy.time_provenance_degraded_behavior,
            )
    return defaults


def _iter_source_paths(feature_key: str, meta_json: Mapping[str, Any]) -> List[Tuple[str, str, str]]:
    out: List[Tuple[str, str, str]] = []
    for src in meta_json.get("source_endpoints", []) or []:
        if not isinstance(src, Mapping):
            continue
        path = src.get("path")
        if not path:
            continue
        method = str(src.get("method") or "GET")
        out.append((method, str(path), "source_endpoints"))
    lineage = meta_json.get("metric_lineage", {}) or {}
    source_path = lineage.get("source_path")
    if source_path:
        out.append(("GET", str(source_path), "metric_lineage"))
    if not out:
        fallback_name = _FEATURE_KEY_FALLBACK_POLICIES.get(feature_key)
        if fallback_name:
            out.append(("GET", fallback_name, "feature_key_default"))
    return out


def resolve_feature_policy(feature_key: str, meta_json: Mapping[str, Any], cfg: Mapping[str, Any]) -> ResolvedFeaturePolicy:
    defaults = default_endpoint_policy(cfg)
    sources = _iter_source_paths(feature_key, meta_json)
    resolved: List[Tuple[EndpointFreshnessPolicy, str]] = []
    for method, path, source_kind in sources:
        if path == "generic_default":
            resolved.append((defaults, source_kind))
            continue
        resolved.append((resolve_endpoint_policy(method, path, cfg), source_kind))

    if not resolved:
        return ResolvedFeaturePolicy(
            name=defaults.name,
            max_tolerated_age_seconds=defaults.max_tolerated_age_seconds,
            join_skew_tolerance_seconds=defaults.join_skew_tolerance_seconds,
            criticality=defaults.criticality,
            lag_class=defaults.lag_class,
            fresh_behavior=defaults.fresh_behavior,
            stale_behavior=defaults.stale_behavior,
            carry_forward_behavior=defaults.carry_forward_behavior,
            empty_valid_behavior=defaults.empty_valid_behavior,
            time_provenance_degraded_behavior=defaults.time_provenance_degraded_behavior,
            sources=(),
        )

    policies = [p for p, _ in resolved]
    return ResolvedFeaturePolicy(
        name="+".join(sorted({p.name for p in policies})),
        max_tolerated_age_seconds=min(p.max_tolerated_age_seconds for p in policies),
        join_skew_tolerance_seconds=min(p.join_skew_tolerance_seconds for p in policies),
        criticality=(EndpointCriticality.CRITICAL if any(p.criticality == EndpointCriticality.CRITICAL for p in policies) else EndpointCriticality.NON_CRITICAL),
        lag_class=_merge_lag_class(p.lag_class for p in policies),
        fresh_behavior=_merge_action((p.fresh_behavior for p in policies), defaults.fresh_behavior),
        stale_behavior=_merge_action((p.stale_behavior for p in policies), defaults.stale_behavior),
        carry_forward_behavior=_merge_action((p.carry_forward_behavior for p in policies), defaults.carry_forward_behavior),
        empty_valid_behavior=_merge_action((p.empty_valid_behavior for p in policies), defaults.empty_valid_behavior),
        time_provenance_degraded_behavior=_merge_action((p.time_provenance_degraded_behavior for p in policies), defaults.time_provenance_degraded_behavior),
        sources=tuple(sorted({f"{source_kind}:{p.name}" for p, source_kind in resolved})),
    )


def _parse_effective_ts(eff_ts_raw: Any) -> Tuple[Optional[dt.datetime], Optional[FeatureDecisionReason], Optional[str]]:
    if not isinstance(eff_ts_raw, str) or not eff_ts_raw:
        return None, FeatureDecisionReason.MISSING_EFFECTIVE_TS, None
    try:
        eff_ts = dt.datetime.fromisoformat(eff_ts_raw.replace("Z", "+00:00"))
    except Exception:
        return None, FeatureDecisionReason.INVALID_EFFECTIVE_TS, "malformed"
    if eff_ts.tzinfo is None:
        return None, FeatureDecisionReason.INVALID_EFFECTIVE_TS, "naive_timezone"
    return eff_ts.astimezone(dt.timezone.utc), None, None


def _value_is_valid(feature_value: Any) -> bool:
    return isinstance(feature_value, (int, float)) and math.isfinite(float(feature_value))


def assess_feature_freshness(
    feature: Mapping[str, Any],
    *,
    asof_utc: dt.datetime,
    cadence_seconds: int,
    cfg: Mapping[str, Any],
) -> FeaturePolicyAssessment:
    feature_key = str(feature.get("feature_key"))
    meta_json = feature.get("meta_json", {}) or {}
    policy = resolve_feature_policy(feature_key, meta_json, cfg)
    metric_lineage = meta_json.get("metric_lineage", {}) or {}
    feature_value = feature.get("feature_value")
    freshness_state = str(meta_json.get("freshness_state") or "ERROR")
    stale_age_minutes = meta_json.get("stale_age_min")
    stale_age_seconds = None if stale_age_minutes is None else _clamp_positive_int(stale_age_minutes, 0) * 60
    time_provenance_degraded = bool(metric_lineage.get("time_provenance_degraded") or meta_json.get("details", {}).get("time_provenance_degraded"))

    eff_ts, ts_error, ts_error_detail = _parse_effective_ts(metric_lineage.get("effective_ts_utc"))
    if ts_error is not None:
        return FeaturePolicyAssessment(
            feature_key=feature_key,
            policy=policy,
            include_in_scoring=False,
            degraded=False,
            reason=ts_error,
            reason_detail=ts_error_detail,
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

    # Allow a small amount of provider-time forward drift to be normalized/clamped.
    #
    # Why this exists: in live operation, host clocks can drift a few minutes behind the
    # vendor server time (especially on non-NTP-synced desktops). Without an explicit
    # tolerance, this manifests as FUTURE_TS_VIOLATION and collapses feature coverage.
    #
    # Contract:
    # - INSTITUTIONAL_GRADE: do not implicitly widen the window (strict-by-default)
    # - FORWARD_OBSERVATION: apply a pragmatic default floor when not explicitly configured
    allow_future_ts_seconds = cadence_seconds
    try:
        validation_cfg = cfg.get("validation") if isinstance(cfg, dict) else None
        governance_mode = "FORWARD_OBSERVATION"
        if isinstance(validation_cfg, dict):
            gm = validation_cfg.get("governance_mode")
            if gm is not None:
                governance_mode = str(gm).upper().strip()

            if validation_cfg.get("allow_future_ts_seconds") is not None:
                allow_future_ts_seconds = max(int(validation_cfg.get("allow_future_ts_seconds")), 0)
            elif governance_mode != "INSTITUTIONAL_GRADE":
                # Default floor for forward/paper observation: tolerate typical host clock skew.
                allow_future_ts_seconds = max(allow_future_ts_seconds, 600)
        else:
            # No validation config available; choose a conservative but usable default.
            allow_future_ts_seconds = max(allow_future_ts_seconds, 600)
    except Exception:
        allow_future_ts_seconds = max(cadence_seconds, 600)

    normalized_future_ts = False
    future_drift_seconds: Optional[int] = None
    future_ts_degraded = False
    delta_seconds = int((asof_utc - eff_ts).total_seconds())
    if delta_seconds < 0:
        drift = abs(delta_seconds)
        if drift <= allow_future_ts_seconds:
            # Clamp to asof_utc so downstream join-skew/staleness calculations remain conservative.
            future_drift_seconds = drift
            future_ts_degraded = drift > cadence_seconds
            eff_ts = asof_utc
            delta_seconds = 0
            normalized_future_ts = True
            if drift > 0:
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

    if freshness_state == "STALE_CARRY" and stale_age_seconds is not None and stale_age_seconds > policy.max_tolerated_age_seconds:
        return FeaturePolicyAssessment(
            feature_key=feature_key,
            policy=policy,
            include_in_scoring=False,
            degraded=False,
            reason=FeatureDecisionReason.STALE_ENDPOINT_REJECTED,
            reason_detail=f"age_{stale_age_seconds}s_gt_{policy.max_tolerated_age_seconds}s",
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
        )

    degraded = False
    dq_reason = None

    if freshness_state in ("FRESH",):
        action = policy.fresh_behavior
    elif freshness_state == "EMPTY_VALID":
        action = policy.empty_valid_behavior
    elif freshness_state == "STALE_CARRY":
        age = stale_age_seconds
        if age is not None and age > policy.max_tolerated_age_seconds:
            return FeaturePolicyAssessment(
                feature_key=feature_key,
                policy=policy,
                include_in_scoring=False,
                degraded=False,
                reason=FeatureDecisionReason.STALE_ENDPOINT_REJECTED,
                reason_detail=f"age_{age}s_gt_{policy.max_tolerated_age_seconds}s",
                dq_reason_code=f"{feature_key}_{FeatureDecisionReason.STALE_ENDPOINT_REJECTED.value}",
                effective_ts=eff_ts,
                delta_seconds=delta_seconds,
                normalized_future_ts=normalized_future_ts,
                freshness_state=freshness_state,
                stale_age_seconds=age,
                stale_age_minutes=stale_age_minutes if isinstance(stale_age_minutes, int) else None,
                time_provenance_degraded=time_provenance_degraded,
                policy_source=",".join(policy.sources) if policy.sources else "default",
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
        )

    if time_provenance_degraded and policy.time_provenance_degraded_behavior == PolicyAction.DEGRADE:
        degraded = True
        dq_reason = dq_reason or f"{feature_key}_time_provenance_degraded"

    # If we had to clamp a timestamp that was materially ahead of asof_utc (beyond cadence),
    # keep the feature but mark it degraded for downstream confidence/coverage logic.
    if future_ts_degraded and future_drift_seconds is not None:
        degraded = True
        dq_reason = dq_reason or f"{feature_key}_future_ts_clamped_{future_drift_seconds}s"

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
