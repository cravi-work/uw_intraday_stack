
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from .config_loader import resolve_ood_assessment_policy
from .models import DataQualityState, DecisionGate, OODState, SessionState

_DEFAULT_POLICY: Dict[str, float] = {
    "degraded_coverage_threshold": 0.85,
    "out_coverage_threshold": 0.50,
    "boundary_slack": 1e-6,
}
_BOUNDED_OUTPUT_DOMAINS = {
    "closed_interval",
    "discrete_sign",
    "discrete_enum",
    "bounded_ratio",
    "percentile_rank",
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
    output_domain_missing_features: Tuple[str, ...] = field(default_factory=tuple)
    output_domain_contract_issues: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    contract_version: str = "ood/v2"
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
            "output_domain_missing_features": list(self.output_domain_missing_features),
            "output_domain_contract_issues": self.output_domain_contract_issues,
            "contract_version": self.contract_version,
            "assessment_ran": bool(self.assessment_ran),
        }


@dataclass(frozen=True)
class OutputDomainContract:
    bounded_output: bool
    expected_bounds: Optional[Dict[str, Any]]
    allowed_values: Tuple[float, ...]
    emitted_units: Optional[str]
    raw_input_units: Optional[str]
    output_domain: Optional[str]
    output_domain_contract_version: Optional[str]
    missing_fields: Tuple[str, ...]
    boundary_source: Optional[str]


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


def _safe_float(value: Any) -> Optional[float]:
    try:
        coerced = float(value)
    except (TypeError, ValueError):
        return None
    return coerced if math.isfinite(coerced) else None


def _text_or_none(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_expected_bounds(value: Any) -> Optional[Dict[str, Any]]:
    if value is None:
        return None

    lower: Optional[float] = None
    upper: Optional[float] = None
    inclusive_default = True
    inclusive_lower = True
    inclusive_upper = True

    if isinstance(value, Mapping):
        lower = _safe_float(value.get("lower", value.get("min")))
        upper = _safe_float(value.get("upper", value.get("max")))
        inclusive_default = bool(value.get("inclusive", True))
        inclusive_lower = bool(value.get("inclusive_lower", inclusive_default))
        inclusive_upper = bool(value.get("inclusive_upper", inclusive_default))
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        lower = _safe_float(value[0])
        upper = _safe_float(value[1])
    else:
        return None

    if lower is None or upper is None:
        return None

    lo = float(min(lower, upper))
    hi = float(max(lower, upper))
    return {
        "lower": lo,
        "upper": hi,
        "inclusive": inclusive_default,
        "inclusive_lower": inclusive_lower,
        "inclusive_upper": inclusive_upper,
    }


def _normalize_allowed_values(value: Any) -> Tuple[float, ...]:
    if not isinstance(value, (list, tuple, set, frozenset)):
        return tuple()

    normalized: list[float] = []
    for raw in value:
        numeric = _safe_float(raw)
        if numeric is None:
            return tuple()
        normalized.append(float(numeric))
    return tuple(normalized)


def _extract_output_domain_contract(lineage: Mapping[str, Any]) -> OutputDomainContract:
    expected_bounds = _normalize_expected_bounds(lineage.get("expected_bounds"))
    allowed_values = _normalize_allowed_values(lineage.get("allowed_values"))
    emitted_units = _text_or_none(lineage.get("emitted_units"))
    raw_input_units = _text_or_none(lineage.get("raw_input_units"))
    output_domain = _text_or_none(lineage.get("output_domain"))
    contract_version = _text_or_none(lineage.get("output_domain_contract_version"))

    bounded_output = bool(lineage.get("bounded_output", False))
    if not bounded_output and (
        expected_bounds is not None
        or bool(allowed_values)
        or (output_domain or "").lower() in _BOUNDED_OUTPUT_DOMAINS
    ):
        bounded_output = True

    missing_fields: list[str] = []
    if bounded_output:
        if expected_bounds is None:
            missing_fields.append("expected_bounds")
        if emitted_units is None:
            missing_fields.append("emitted_units")
        if contract_version is None:
            missing_fields.append("output_domain_contract_version")

    boundary_source = "structured" if (expected_bounds is not None or allowed_values) else None
    return OutputDomainContract(
        bounded_output=bounded_output,
        expected_bounds=expected_bounds,
        allowed_values=allowed_values,
        emitted_units=emitted_units,
        raw_input_units=raw_input_units,
        output_domain=output_domain,
        output_domain_contract_version=contract_version,
        missing_fields=tuple(missing_fields),
        boundary_source=boundary_source,
    )


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


def _value_matches_allowed(value: float, allowed: Sequence[float], slack: float) -> bool:
    return any(abs(float(candidate) - value) <= slack for candidate in allowed)


def _value_within_bounds(value: float, bounds: Mapping[str, Any], slack: float) -> bool:
    lower = float(bounds["lower"])
    upper = float(bounds["upper"])
    inclusive_lower = bool(bounds.get("inclusive_lower", bounds.get("inclusive", True)))
    inclusive_upper = bool(bounds.get("inclusive_upper", bounds.get("inclusive", True)))

    lower_ok = value >= (lower - slack) if inclusive_lower else value > (lower + slack)
    upper_ok = value <= (upper + slack) if inclusive_upper else value < (upper - slack)
    return lower_ok and upper_ok


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
    output_domain_contract_issues: Dict[str, Dict[str, Any]] = {}
    degraded_feature_keys: list[str] = []
    session_mismatch_features: list[str] = []
    normalization_failure_features: list[str] = []
    missing_feature_keys: list[str] = []
    output_domain_missing_features: list[str] = []
    valid_decision_feature_count = 0
    slack = float(effective_policy["boundary_slack"])

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

        numeric_value = _safe_float(value)
        if numeric_value is None:
            missing_feature_keys.append(feature_key)
            reasons.append(f"missing_or_invalid_value:{feature_key}")
            if assessment is not None and (
                getattr(assessment, "degraded", False)
                or getattr(assessment, "time_provenance_degraded", False)
            ):
                degraded_feature_keys.append(feature_key)
            continue

        valid_decision_feature_count += 1

        domain_contract = _extract_output_domain_contract(lineage)
        if domain_contract.bounded_output and domain_contract.missing_fields:
            output_domain_missing_features.append(feature_key)
            output_domain_contract_issues[feature_key] = {
                "missing_fields": list(domain_contract.missing_fields),
                "emitted_units": domain_contract.emitted_units,
                "raw_input_units": domain_contract.raw_input_units,
                "output_domain": domain_contract.output_domain,
                "output_domain_contract_version": domain_contract.output_domain_contract_version,
                "units_expected": lineage.get("units_expected"),
            }
            degraded_feature_keys.append(feature_key)
            reasons.append(
                f"output_domain_contract_missing:{feature_key}:{','.join(domain_contract.missing_fields)}"
            )

        violation_payload: Optional[Dict[str, Any]] = None
        if domain_contract.expected_bounds is not None and not _value_within_bounds(numeric_value, domain_contract.expected_bounds, slack):
            violation_payload = {
                "value": float(numeric_value),
                "lower_bound": float(domain_contract.expected_bounds["lower"]),
                "upper_bound": float(domain_contract.expected_bounds["upper"]),
                "inclusive": bool(domain_contract.expected_bounds.get("inclusive", True)),
                "inclusive_lower": bool(domain_contract.expected_bounds.get("inclusive_lower", domain_contract.expected_bounds.get("inclusive", True))),
                "inclusive_upper": bool(domain_contract.expected_bounds.get("inclusive_upper", domain_contract.expected_bounds.get("inclusive", True))),
                "boundary_source": domain_contract.boundary_source,
                "emitted_units": domain_contract.emitted_units,
                "output_domain": domain_contract.output_domain,
                "output_domain_contract_version": domain_contract.output_domain_contract_version,
                "violation_type": "expected_bounds",
            }
        if domain_contract.allowed_values and not _value_matches_allowed(numeric_value, domain_contract.allowed_values, slack):
            allowed_violation = {
                "value": float(numeric_value),
                "allowed_values": [float(candidate) for candidate in domain_contract.allowed_values],
                "boundary_source": domain_contract.boundary_source,
                "emitted_units": domain_contract.emitted_units,
                "output_domain": domain_contract.output_domain,
                "output_domain_contract_version": domain_contract.output_domain_contract_version,
                "violation_type": "allowed_values",
            }
            if violation_payload is None:
                violation_payload = allowed_violation
            else:
                violation_payload["allowed_values"] = allowed_violation["allowed_values"]
                violation_payload["violation_type"] = "expected_bounds_and_allowed_values"

        if violation_payload is not None:
            boundary_violations[feature_key] = violation_payload
            reasons.append(f"boundary_violation:{feature_key}")

        if bool(lineage.get("time_provenance_degraded")):
            degraded_feature_keys.append(feature_key)
            reasons.append(f"time_provenance_degraded:{feature_key}")

        if assessment is not None and (
            getattr(assessment, "degraded", False)
            or getattr(assessment, "time_provenance_degraded", False)
        ):
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
    elif (
        output_domain_missing_features
        or session_mismatch_features
        or degraded_feature_keys
        or coverage_ratio < float(effective_policy["degraded_coverage_threshold"])
    ):
        state = OODState.DEGRADED
        if output_domain_missing_features:
            primary_reason = next(r for r in unique_reasons if r.startswith("output_domain_contract_missing:"))
        elif session_mismatch_features:
            primary_reason = next(r for r in unique_reasons if r.startswith("session_mismatch:"))
        elif degraded_feature_keys:
            primary_reason = next(
                (
                    r
                    for r in unique_reasons
                    if r.startswith("time_provenance_degraded:") or r.startswith("freshness_degraded:")
                ),
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
        output_domain_missing_features=tuple(sorted(set(output_domain_missing_features))),
        output_domain_contract_issues=output_domain_contract_issues,
    )
