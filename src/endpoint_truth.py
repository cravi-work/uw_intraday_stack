from __future__ import annotations
from enum import Enum
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
import datetime

from .endpoint_rules import EmptyPayloadPolicy, get_empty_policy

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

@dataclass(frozen=True)
class PayloadAssessment:
    payload_class: EndpointPayloadClass
    empty_policy: EmptyPayloadPolicy
    is_empty: bool
    changed: Optional[bool]
    error_reason: Optional[str]

@dataclass
class MetaContract:
    source_endpoints: List[Dict[str, Any]]
    freshness_state: str
    stale_age_min: Optional[int]
    na_reason: Optional[str]

@dataclass
class ResolvedLineage:
    used_event_id: Optional[str]
    freshness_state: FreshnessState
    stale_age_seconds: Optional[int]
    payload_class: EndpointPayloadClass
    na_reason: Optional[str]

def to_utc_dt(x: Any, *, fallback: datetime.datetime) -> datetime.datetime:
    """Deterministically normalizes any timestamp to a UTC aware datetime."""
    if x is None:
        return fallback
    if isinstance(x, (int, float)):
        return datetime.datetime.fromtimestamp(x, datetime.timezone.utc)
    if isinstance(x, datetime.datetime):
        if x.tzinfo is None:
            return x.replace(tzinfo=datetime.timezone.utc)
        return x.astimezone(datetime.timezone.utc)
    return fallback

def classify_payload(res: Any, prev_hash: Optional[str], method: str, path: str, session_label: str) -> PayloadAssessment:
    if not getattr(res, "ok", False) or getattr(res, "payload_json", None) is None:
        return PayloadAssessment(EndpointPayloadClass.ERROR, EmptyPayloadPolicy.EMPTY_INVALID, False, None, getattr(res, "error_message", "HTTP Error or Missing JSON"))

    pj = res.payload_json
    is_empty = False
    if isinstance(pj, dict) and not pj: is_empty = True
    if isinstance(pj, list) and not pj: is_empty = True
    
    # Unwrap UW specific envelope schemas
    if isinstance(pj, dict) and len(pj) == 1:
        if "data" in pj and not pj["data"]: is_empty = True
        if "results" in pj and not pj["results"]: is_empty = True

    policy = get_empty_policy(method, path, session_label)

    if is_empty:
        if policy != EmptyPayloadPolicy.EMPTY_INVALID:
            return PayloadAssessment(EndpointPayloadClass.SUCCESS_EMPTY_VALID, policy, True, False, None)
        return PayloadAssessment(EndpointPayloadClass.ERROR, policy, True, None, "EMPTY_UNEXPECTED")

    changed = (res.payload_hash != prev_hash)
    pclass = EndpointPayloadClass.SUCCESS_HAS_DATA if changed else EndpointPayloadClass.SUCCESS_STALE
    return PayloadAssessment(pclass, policy, False, changed, None)

def resolve_effective_payload(
    current_event_id: str, 
    current_ts: datetime.datetime, 
    assessment: PayloadAssessment, 
    prev_state: Optional[Dict[str, Any]]
) -> ResolvedLineage:
    
    pclass = assessment.payload_class

    if pclass == EndpointPayloadClass.SUCCESS_HAS_DATA:
        return ResolvedLineage(current_event_id, FreshnessState.FRESH, 0, pclass, None)
    
    if pclass == EndpointPayloadClass.SUCCESS_STALE:
        used_id = prev_state["last_change_event_id"] if prev_state and prev_state.get("last_change_event_id") else current_event_id
        prev_change_ts = to_utc_dt(prev_state.get("last_change_ts_utc"), fallback=current_ts) if prev_state else current_ts
        age = max(0, int((current_ts - prev_change_ts).total_seconds()))
        return ResolvedLineage(used_id, FreshnessState.STALE_CARRY, age, pclass, None)

    if pclass == EndpointPayloadClass.SUCCESS_EMPTY_VALID:
        if assessment.empty_policy == EmptyPayloadPolicy.EMPTY_IS_DATA:
            return ResolvedLineage(current_event_id, FreshnessState.EMPTY_VALID, 0, pclass, None)
        elif assessment.empty_policy == EmptyPayloadPolicy.EMPTY_MEANS_STALE:
            if prev_state and prev_state.get("last_success_event_id"):
                prev_success_ts = to_utc_dt(prev_state.get("last_success_ts_utc"), fallback=current_ts)
                age = max(0, int((current_ts - prev_success_ts).total_seconds()))
                return ResolvedLineage(prev_state["last_success_event_id"], FreshnessState.STALE_CARRY, age, pclass, f"CARRY_FORWARD_{pclass.name}")
            else:
                return ResolvedLineage(None, FreshnessState.EMPTY_VALID, None, pclass, "NO_PRIOR_SUCCESS")

    # ERROR
    if prev_state and prev_state.get("last_success_event_id"):
        prev_success_ts = to_utc_dt(prev_state.get("last_success_ts_utc"), fallback=current_ts)
        age = max(0, int((current_ts - prev_success_ts).total_seconds()))
        return ResolvedLineage(prev_state["last_success_event_id"], FreshnessState.STALE_CARRY, age, pclass, "CARRY_FORWARD_ERROR")
    
    return ResolvedLineage(None, FreshnessState.ERROR, None, pclass, "NO_PRIOR_SUCCESS")