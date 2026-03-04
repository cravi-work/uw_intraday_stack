from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from .config_loader import resolve_ood_assessment_policy
from .models import DataQualityState, DecisionGate, OODState, SessionState

_BOUND_RE = re.compile(r"\[\s*([-+−]?\d+(?:\.\d+)?)\s*,\s*([-+−]?\d+(?:\.\d+)?)\s*\]")
_DEFAULT_POLICY: Dict[str, float] = {
    "degraded_coverage_threshold": 0.85,
    "out_coverage_threshold": 0.50,
    "boundary_slack": 1e-6,
}


@dataclass(frozen=True)
class OODAssessment:
    state: OODState
    primary_reason: str
    reasons: Tuple[str, ...]
    decision_feature_keys: Tuple[str, ...]
    valid_decision_feature_count: int
    total_decision_feature_count: int
    coverage_ratio: float
    session_state: str
    degraded_feature_keys: Tuple[str, ...] = field(default_factory=tuple)
    session_mismatch_features: Tuple[str, ...] = field(default_factory=tuple)
    normalization_failure_features: Tuple[str, ...] = field(default_factory=tuple)
    missing_feature_keys: Tuple[str, ...] = field(default_factory=tuple)
    boundary_violation_features: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    contract_version: str = "ood/v1"
    assessment_ran: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state.value,
            "primary_reason": self.primary_reason,
            "reasons": list(self.reasons),
            "decision_feature_keys": list(self.decision_feature_keys),
            "valid_decision_feature_count": int(self.valid_decision_feature_count),
            "total_decision_feature_count": int(self.total_decision_feature_count),
            "coverage_ratio": float(self.coverage_ratio),
            "session_state": self.session_state,
            "degraded_feature_keys": list(self.degraded_feature_keys),
            "session_mismatch_features": list(self.session_mismatch_features),
            "normalization_failure_features": list(self.normalization_failure_features),
            "missing_feature_keys": list(self.missing_feature_keys),
            "boundary_violation_features": self.boundary_violation_features,
            "contract_version": self.contract_version,
            "assessment_ran": bool(self.assessment_ran),
        }


def _mapping(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _coerce_session_state(value: Any) -> str:
    if isinstance(value, SessionState):
        return value.value
    raw = str(value or "").strip().upper()
    return raw if raw else SessionState.CLOSED.value


def resolve_ood_policy(cfg: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    policy: Dict[str, Any] = dict(_DEFAULT_POLICY)
    if not isinstance(cfg, Mapping):
        return policy

    resolved = resolve_ood_assessment_policy(cfg)
    if isinstance(resolved, Mapping):
        for key in ("contract_version", "require_assessment_before_emission"):
            if key in resolved:
                policy[key] = resolved[key]
        for key in list(_DEFAULT_POLICY.keys()):
            if key in resolved:
                try:
                    policy[key] = float(resolved[key])
                except (TypeError, ValueError):
                    continue
    return policy


def _parse_expected_bounds(units_expected: Any) -> Optional[Tuple[float, float]]:
    if not isinstance(units_expected, str) or not units_expected.strip():
        return None
    match = _BOUND_RE.search(units_expected.replace("−", "-"))
    if not match:
        return None
    left = float(match.group(1))
    right = float(match.group(2))
    lo = min(left, right)
    hi = max(left, right)
    return lo, hi


def _session_allowed(session_applicability: Any, session_state: str) -> bool:
    if not isinstance(session_applicability, str) or not session_applicability.strip():
        return True
    normalized = session_applicability.replace(",", "/").replace(" ", "")
    tokens = {token.upper() for token in normalized.split("/") if token}
    if not tokens:
        return True
    if {"ALL", "ANY"} & tokens:
        return True
    return session_state.upper() in tokens


def _contains_normalization_failure(meta: Mapping[str, Any]) -> bool:
    na_reason = str(meta.get("na_reason") or "").lower()
    if "normalization" in na_reason:
        return True
    details = _mapping(meta.get("details"))
    norm = _mapping(details.get("contract_normalization"))
    if not norm:
        norm = _mapping(_mapping(meta.get("metric_lineage")).get("contract_normalization"))
    status = str(norm.get("status") or "").lower()
    failure_reason = str(norm.get("failure_reason") or "").lower()
    return any(
        token in status or token in failure_reason
        for token in ("invalid", "failure", "mismatch", "unparsed", "suppressed")
    )


def assess_operational_ood(
    *,
    feature_rows: Sequence[Mapping[str, Any]],
    decision_feature_keys: Sequence[str],
    session_state: Any,
    assessments_by_feature: Optional[Mapping[str, Any]] = None,
    gate: Optional[DecisionGate] = None,
    policy: Optional[Mapping[str, Any]] = None,
) -> OODAssessment:
    resolved_session = _coerce_session_state(session_state)
    decision_keys = tuple(sorted({str(key) for key in decision_feature_keys if str(key)}))

    if resolved_session == SessionState.CLOSED.value:
        return OODAssessment(
            state=OODState.UNKNOWN,
            primary_reason="session_closed_assessment_skipped",
            reasons=("session_closed_assessment_skipped",),
            decision_feature_keys=decision_keys,
            valid_decision_feature_count=0,
            total_decision_feature_count=len(decision_keys),
            coverage_ratio=0.0,
            session_state=resolved_session,
            assessment_ran=False,
        )

    if not decision_keys:
        return OODAssessment(
            state=OODState.UNKNOWN,
            primary_reason="no_decision_features_configured",
            reasons=("no_decision_features_configured",),
            decision_feature_keys=decision_keys,
            valid_decision_feature_count=0,
            total_decision_feature_count=0,
            coverage_ratio=0.0,
            session_state=resolved_session,
            assessment_ran=False,
        )

    effective_policy = dict(_DEFAULT_POLICY)
    if isinstance(policy, Mapping):
        for key, fallback in _DEFAULT_POLICY.items():
            try:
                effective_policy[key] = float(policy.get(key, fallback))
            except (TypeError, ValueError):
                effective_policy[key] = float(fallback)

    assessments = dict(assessments_by_feature or {})
    rows_by_key: Dict[str, Mapping[str, Any]] = {
        str(row.get("feature_key")): row
        for row in feature_rows
        if isinstance(row, Mapping) and row.get("feature_key") is not None
    }

    reasons: list[str] = []
    boundary_violations: Dict[str, Dict[str, Any]] = {}
    degraded_feature_keys: list[str] = []
    session_mismatch_features: list[str] = []
    normalization_failure_features: list[str] = []
    missing_feature_keys: list[str] = []
    valid_decision_feature_count = 0

    for feature_key in decision_keys:
        row = rows_by_key.get(feature_key)
        assessment = assessments.get(feature_key)
        meta = _mapping(row.get("meta_json")) if row is not None else {}
        lineage = _mapping(meta.get("metric_lineage"))
        value = row.get("feature_value") if row is not None else None

        if row is None:
            missing_feature_keys.append(feature_key)
            reasons.append(f"missing_feature:{feature_key}")
            continue

        if _contains_normalization_failure(meta):
            normalization_failure_features.append(feature_key)
            reasons.append(f"normalization_failure:{feature_key}")

        if not _session_allowed(lineage.get("session_applicability"), resolved_session):
            session_mismatch_features.append(feature_key)
            reasons.append(f"session_mismatch:{feature_key}")

        if assessment is not None and getattr(assessment, "include_in_scoring", True) is False:
            missing_feature_keys.append(feature_key)
            detail = getattr(assessment, "reason", None)
            if detail is not None:
                reasons.append(f"freshness_excluded:{feature_key}:{getattr(detail, 'value', detail)}")
            if getattr(assessment, "degraded", False) or getattr(assessment, "time_provenance_degraded", False):
                degraded_feature_keys.append(feature_key)
            continue

        finite_value = value is not None and math.isfinite(float(value))
        if not finite_value:
            missing_feature_keys.append(feature_key)
            reasons.append(f"missing_or_invalid_value:{feature_key}")
            if assessment is not None and (getattr(assessment, "degraded", False) or getattr(assessment, "time_provenance_degraded", False)):
                degraded_feature_keys.append(feature_key)
            continue

        valid_decision_feature_count += 1

        bounds = _parse_expected_bounds(lineage.get("units_expected"))
        if bounds is not None:
            lo, hi = bounds
            slack = float(effective_policy["boundary_slack"])
            if float(value) < (lo - slack) or float(value) > (hi + slack):
                boundary_violations[feature_key] = {
                    "value": float(value),
                    "lower_bound": float(lo),
                    "upper_bound": float(hi),
                    "units_expected": lineage.get("units_expected"),
                }
                reasons.append(f"boundary_violation:{feature_key}")

        if bool(lineage.get("time_provenance_degraded")):
            degraded_feature_keys.append(feature_key)
            reasons.append(f"time_provenance_degraded:{feature_key}")

        if assessment is not None and (getattr(assessment, "degraded", False) or getattr(assessment, "time_provenance_degraded", False)):
            degraded_feature_keys.append(feature_key)
            detail = getattr(assessment, "reason", None)
            reasons.append(f"freshness_degraded:{feature_key}:{getattr(detail, 'value', detail)}")

    total = len(decision_keys)
    coverage_ratio = float(valid_decision_feature_count / total) if total > 0 else 0.0

    if gate is not None:
        if gate.risk_gate_status.value == "DEGRADED":
            reasons.append("gate_degraded")
        if gate.data_quality_state == DataQualityState.PARTIAL:
            reasons.append("gate_partial_quality")
        if gate.data_quality_state == DataQualityState.INVALID and gate.risk_gate_status.value != "BLOCKED":
            reasons.append("gate_invalid_quality")

    unique_reasons = tuple(dict.fromkeys(reasons))

    if boundary_violations:
        state = OODState.OUT_OF_DISTRIBUTION
        primary_reason = next(r for r in unique_reasons if r.startswith("boundary_violation:"))
    elif normalization_failure_features:
        state = OODState.OUT_OF_DISTRIBUTION
        primary_reason = next(r for r in unique_reasons if r.startswith("normalization_failure:"))
    elif coverage_ratio < float(effective_policy["out_coverage_threshold"]):
        state = OODState.OUT_OF_DISTRIBUTION
        primary_reason = f"coverage_below_out_threshold:{coverage_ratio:.2f}"
    elif session_mismatch_features or degraded_feature_keys or coverage_ratio < float(effective_policy["degraded_coverage_threshold"]):
        state = OODState.DEGRADED
        if session_mismatch_features:
            primary_reason = next(r for r in unique_reasons if r.startswith("session_mismatch:"))
        elif degraded_feature_keys:
            primary_reason = next(
                (r for r in unique_reasons if r.startswith("time_provenance_degraded:") or r.startswith("freshness_degraded:")),
                f"coverage_below_degraded_threshold:{coverage_ratio:.2f}",
            )
        else:
            primary_reason = f"coverage_below_degraded_threshold:{coverage_ratio:.2f}"
    elif not unique_reasons:
        state = OODState.IN_DISTRIBUTION
        primary_reason = "decision_feature_bundle_in_distribution"
        unique_reasons = (primary_reason,)
    else:
        state = OODState.IN_DISTRIBUTION
        primary_reason = unique_reasons[0]

    return OODAssessment(
        state=state,
        primary_reason=primary_reason,
        reasons=unique_reasons,
        decision_feature_keys=decision_keys,
        valid_decision_feature_count=valid_decision_feature_count,
        total_decision_feature_count=total,
        coverage_ratio=coverage_ratio,
        session_state=resolved_session,
        degraded_feature_keys=tuple(sorted(set(degraded_feature_keys))),
        session_mismatch_features=tuple(sorted(set(session_mismatch_features))),
        normalization_failure_features=tuple(sorted(set(normalization_failure_features))),
        missing_feature_keys=tuple(sorted(set(missing_feature_keys))),
        boundary_violation_features=boundary_violations,
    )
