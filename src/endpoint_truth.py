from __future__ import annotations
from enum import Enum
from dataclasses import dataclass, replace
from typing import Optional, List, Dict, Any, Tuple
import datetime

from .endpoint_rules import EmptyPayloadPolicy, EndpointRule, get_empty_policy, get_endpoint_rule

class EndpointPayloadClass(Enum):
    SUCCESS_HAS_DATA = "SUCCESS_HAS_DATA"
    SUCCESS_EMPTY_VALID = "SUCCESS_EMPTY_VALID"
    SUCCESS_STALE = "SUCCESS_STALE"
    ERROR = "ERROR"

class FreshnessState(Enum):
    FRESH = "FRESH"
    STALE_CARRY = "STALE_CARRY"
    EMPTY_VALID = "EMPTY_VALID"
    ERROR = "ERROR"

class NaReasonCode(str, Enum):
    NO_PRIOR_SUCCESS = "NO_PRIOR_SUCCESS"
    CARRY_FORWARD_ERROR = "CARRY_FORWARD_ERROR"
    CARRY_FORWARD_EMPTY_MEANS_STALE = "CARRY_FORWARD_EMPTY_MEANS_STALE"
    DEPENDENCY_ERROR = "DEPENDENCY_ERROR"
    STALE_TOO_OLD = "STALE_TOO_OLD"
    STALE_WARN = "STALE_WARN"
    UNRESOLVED = "UNRESOLVED"
    USED_EVENT_NOT_FOUND = "USED_EVENT_NOT_FOUND"
    PAYLOAD_JSON_INVALID = "PAYLOAD_JSON_INVALID"

@dataclass(frozen=True)
class EndpointStateRow:
    last_success_event_id: Optional[str]
    last_success_ts_utc: Optional[datetime.datetime]
    last_payload_hash: Optional[str]
    last_change_ts_utc: Optional[datetime.datetime]
    last_change_event_id: Optional[str]

@dataclass(frozen=True)
class EndpointContext:
    endpoint_id: int
    method: str
    path: str
    operation_id: Optional[str]
    signature: str
    used_event_id: Optional[str]
    payload_class: str
    freshness_state: str
    stale_age_min: Optional[int]
    na_reason: Optional[str]

@dataclass(frozen=True)
class PayloadAssessment:
    payload_class: EndpointPayloadClass
    empty_policy: EmptyPayloadPolicy
    is_empty: bool
    changed: Optional[bool]
    error_reason: Optional[str]
    missing_keys: Optional[List[str]] = None
    validator: Optional[str] = None

@dataclass
class MetaContract:
    source_endpoints: List[Dict[str, Any]]
    freshness_state: str
    stale_age_min: Optional[int]
    na_reason: Optional[str]
    details: Dict[str, Any]

@dataclass
class ResolvedLineage:
    used_event_id: Optional[str]
    freshness_state: FreshnessState
    stale_age_seconds: Optional[int]
    payload_class: EndpointPayloadClass
    na_reason: Optional[str]

def to_utc_dt(x: Any, *, fallback: datetime.datetime) -> datetime.datetime:
    if x is None: return fallback
    if isinstance(x, (int, float)): return datetime.datetime.fromtimestamp(x, datetime.timezone.utc)
    if isinstance(x, datetime.datetime):
        if x.tzinfo is None: return x.replace(tzinfo=datetime.timezone.utc)
        return x.astimezone(datetime.timezone.utc)
    return fallback

def is_provider_error_envelope(pj: Any) -> bool:
    if not isinstance(pj, dict): return False
    keys = set(pj.keys())
    error_keys = {"error", "message", "detail", "status"}
    data_containers = {"data", "results", "items", "trades", "history"}
    return bool(keys.intersection(error_keys) and not keys.intersection(data_containers))

def validate_shape(pj: Any, rule: Optional[EndpointRule]) -> Tuple[bool, Optional[str], Optional[List[str]]]:
    if not rule or not isinstance(pj, dict): return True, None, None
    actual_pj = pj
    for container in rule.data_container_keys:
        if container in pj and len(pj.keys()) <= 2:
            actual_pj = pj[container]
            break
    if not isinstance(actual_pj, dict): return True, None, None

    keys = set(actual_pj.keys())
    if rule.required_all_keys:
        missing = [k for k in rule.required_all_keys if k not in keys]
        if missing: return False, "INVALID_SHAPE:Missing required_all_keys", missing
    if rule.required_any_keys:
        if not any(k in keys for k in rule.required_any_keys):
            return False, "INVALID_SHAPE:Missing required_any_keys", list(rule.required_any_keys)
    return True, None, None

def _enforce_invariants(res: ResolvedLineage) -> ResolvedLineage:
    if res.freshness_state == FreshnessState.STALE_CARRY and not res.used_event_id:
        return replace(res, used_event_id=None, freshness_state=FreshnessState.ERROR, stale_age_seconds=None, na_reason=NaReasonCode.NO_PRIOR_SUCCESS.value)
    return res

def _apply_stale_cutoff(res: ResolvedLineage, age: int, fallback_max_age: int, invalid_after: int) -> ResolvedLineage:
    if age > invalid_after:
        return replace(res, freshness_state=FreshnessState.ERROR, used_event_id=None, na_reason=NaReasonCode.STALE_TOO_OLD.value)
    elif age > fallback_max_age:
        warn_str = NaReasonCode.STALE_WARN.value
        new_reason = f"{res.na_reason}|{warn_str}" if res.na_reason else warn_str
        return replace(res, na_reason=new_reason)
    return res

def classify_payload(res: Any, prev_hash: Optional[str], method: str, path: str, session_label: str) -> PayloadAssessment:
    rule = get_endpoint_rule(method, path)
    policy = get_empty_policy(method, path, session_label)

    if not getattr(res, "ok", False) or getattr(res, "payload_json", None) is None:
        return PayloadAssessment(EndpointPayloadClass.ERROR, policy, False, None, getattr(res, "error_message", "HTTP Error or Missing JSON"))

    pj = res.payload_json
    if is_provider_error_envelope(pj):
        return PayloadAssessment(EndpointPayloadClass.ERROR, policy, False, None, "PROVIDER_ERROR_ENVELOPE")

    is_empty = False
    if isinstance(pj, dict) and not pj: is_empty = True
    if isinstance(pj, list) and not pj: is_empty = True
    if isinstance(pj, dict) and len(pj) == 1:
        if "data" in pj and not pj["data"]: is_empty = True
        if "results" in pj and not pj["results"]: is_empty = True

    computed_changed = True if prev_hash is None else (res.payload_hash != prev_hash)

    if is_empty:
        if policy == EmptyPayloadPolicy.EMPTY_IS_DATA:
            return PayloadAssessment(EndpointPayloadClass.SUCCESS_EMPTY_VALID, policy, True, computed_changed, None)
        elif policy == EmptyPayloadPolicy.EMPTY_MEANS_STALE:
            return PayloadAssessment(EndpointPayloadClass.SUCCESS_EMPTY_VALID, policy, True, False, "EMPTY_MEANS_STALE")
        else:
            return PayloadAssessment(EndpointPayloadClass.ERROR, policy, True, None, "EMPTY_UNEXPECTED")

    is_valid, shape_err, missing = validate_shape(pj, rule)
    if not is_valid:
        return PayloadAssessment(EndpointPayloadClass.ERROR, policy, False, None, shape_err, missing, "SHAPE_VALIDATOR")

    pclass = EndpointPayloadClass.SUCCESS_HAS_DATA if computed_changed else EndpointPayloadClass.SUCCESS_STALE
    return PayloadAssessment(pclass, policy, False, computed_changed, None)

def _get_carry_forward_target(prev_state: Optional[EndpointStateRow], current_ts: datetime.datetime) -> Tuple[Optional[str], Optional[int]]:
    """Prioritizes the exact timestamp of payload transformation (last_change), otherwise falls back to last fetch."""
    if not prev_state:
        return None, None
    
    if prev_state.last_change_event_id and prev_state.last_change_ts_utc:
        prev_change_ts = to_utc_dt(prev_state.last_change_ts_utc, fallback=current_ts)
        age = max(0, int((current_ts - prev_change_ts).total_seconds()))
        return prev_state.last_change_event_id, age
        
    if prev_state.last_success_event_id and prev_state.last_success_ts_utc:
        prev_success_ts = to_utc_dt(prev_state.last_success_ts_utc, fallback=current_ts)
        age = max(0, int((current_ts - prev_success_ts).total_seconds()))
        return prev_state.last_success_event_id, age
        
    return None, None

def resolve_effective_payload(
    current_event_id: str, current_ts_raw: Any, assessment: PayloadAssessment, 
    prev_state: Optional[EndpointStateRow], fallback_max_age_seconds: int = 900, invalid_after_seconds: int = 3600
) -> ResolvedLineage:
    current_ts = to_utc_dt(current_ts_raw, fallback=datetime.datetime.now(datetime.timezone.utc))
    pclass = assessment.payload_class
    empty_policy = assessment.empty_policy

    if pclass == EndpointPayloadClass.SUCCESS_HAS_DATA:
        return _enforce_invariants(ResolvedLineage(current_event_id, FreshnessState.FRESH, 0, pclass, None))
    
    if pclass == EndpointPayloadClass.SUCCESS_STALE:
        used_id = prev_state.last_change_event_id if prev_state and prev_state.last_change_event_id else current_event_id
        prev_change_ts = to_utc_dt(prev_state.last_change_ts_utc, fallback=current_ts) if prev_state and prev_state.last_change_ts_utc else current_ts
        age = max(0, int((current_ts - prev_change_ts).total_seconds()))
        resolved = ResolvedLineage(used_id, FreshnessState.STALE_CARRY, age, pclass, None)
        return _enforce_invariants(_apply_stale_cutoff(resolved, age, fallback_max_age_seconds, invalid_after_seconds))

    if pclass == EndpointPayloadClass.SUCCESS_EMPTY_VALID:
        if empty_policy == EmptyPayloadPolicy.EMPTY_IS_DATA:
            return _enforce_invariants(ResolvedLineage(current_event_id, FreshnessState.EMPTY_VALID, 0, pclass, None))
        elif empty_policy == EmptyPayloadPolicy.EMPTY_MEANS_STALE:
            cf_id, age = _get_carry_forward_target(prev_state, current_ts)
            if cf_id:
                resolved = ResolvedLineage(cf_id, FreshnessState.STALE_CARRY, age, pclass, NaReasonCode.CARRY_FORWARD_EMPTY_MEANS_STALE.value)
                return _enforce_invariants(_apply_stale_cutoff(resolved, age, fallback_max_age_seconds, invalid_after_seconds))
            else:
                return _enforce_invariants(ResolvedLineage(None, FreshnessState.ERROR, None, pclass, NaReasonCode.NO_PRIOR_SUCCESS.value))

    if pclass == EndpointPayloadClass.ERROR:
        cf_id, age = _get_carry_forward_target(prev_state, current_ts)
        if cf_id:
            resolved = ResolvedLineage(cf_id, FreshnessState.STALE_CARRY, age, pclass, NaReasonCode.CARRY_FORWARD_ERROR.value)
            return _enforce_invariants(_apply_stale_cutoff(resolved, age, fallback_max_age_seconds, invalid_after_seconds))
        else:
            return _enforce_invariants(ResolvedLineage(None, FreshnessState.ERROR, None, pclass, NaReasonCode.NO_PRIOR_SUCCESS.value))
            
    return _enforce_invariants(ResolvedLineage(None, FreshnessState.ERROR, None, pclass, NaReasonCode.UNRESOLVED.value))