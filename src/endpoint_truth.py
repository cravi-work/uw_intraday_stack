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
        return PayloadAssessment(
            payload_class=EndpointPayloadClass.ERROR, 
            empty_policy=EmptyPayloadPolicy.EMPTY_INVALID, 
            is_empty=False, 
            changed=None, 
            error_reason=getattr(res, "error_message", "HTTP Error or Missing JSON")
        )

    pj = res.payload_json
    is_empty = False
    if isinstance(pj, dict) and not pj: is_empty = True
    if isinstance(pj, list) and not pj: is_empty = True
    
    if isinstance(pj, dict) and len(pj) == 1:
        if "data" in pj and not pj["data"]: is_empty = True
        if "results" in pj and not pj["results"]: is_empty = True

    policy = get_empty_policy(method, path, session_label)
    
    # Evaluate changed boolean early so empty-but-valid transitions can trigger a change timestamp update
    computed_changed = True if prev_hash is None else (res.payload_hash != prev_hash)

    if is_empty:
        if policy == EmptyPayloadPolicy.EMPTY_IS_DATA:
            return PayloadAssessment(EndpointPayloadClass.SUCCESS_EMPTY_VALID, policy, True, computed_changed, None)
        elif policy == EmptyPayloadPolicy.EMPTY_MEANS_STALE:
            # changed=False because EMPTY_MEANS_STALE must not overwrite last_success/last_change
            return PayloadAssessment(EndpointPayloadClass.SUCCESS_EMPTY_VALID, policy, True, False, "EMPTY_MEANS_STALE")
        else: # policy == EMPTY_INVALID
            return PayloadAssessment(EndpointPayloadClass.ERROR, policy, True, None, "EMPTY_UNEXPECTED")

    pclass = EndpointPayloadClass.SUCCESS_HAS_DATA if computed_changed else EndpointPayloadClass.SUCCESS_STALE
    
    return PayloadAssessment(
        payload_class=pclass, 
        empty_policy=policy, 
        is_empty=False, 
        changed=computed_changed, 
        error_reason=None
    )

def resolve_effective_payload(
    current_event_id: str, 
    current_ts_raw: Any, 
    assessment: PayloadAssessment, 
    prev_state: Optional[Dict[str, Any]]
) -> ResolvedLineage:
    
    current_ts = to_utc_dt(current_ts_raw, fallback=datetime.datetime.now(datetime.timezone.utc))
    pclass = assessment.payload_class
    empty_policy = assessment.empty_policy

    if pclass == EndpointPayloadClass.SUCCESS_HAS_DATA:
        return ResolvedLineage(
            used_event_id=current_event_id, 
            freshness_state=FreshnessState.FRESH, 
            stale_age_seconds=0, 
            payload_class=pclass, 
            na_reason=None
        )
    
    if pclass == EndpointPayloadClass.SUCCESS_STALE:
        used_id = prev_state["last_change_event_id"] if prev_state and prev_state.get("last_change_event_id") else current_event_id
        prev_change_ts = to_utc_dt(prev_state.get("last_change_ts_utc"), fallback=current_ts) if prev_state else current_ts
        age = max(0, int((current_ts - prev_change_ts).total_seconds()))
        return ResolvedLineage(
            used_event_id=used_id, 
            freshness_state=FreshnessState.STALE_CARRY, 
            stale_age_seconds=age, 
            payload_class=pclass, 
            na_reason=None
        )

    if pclass == EndpointPayloadClass.SUCCESS_EMPTY_VALID:
        if empty_policy == EmptyPayloadPolicy.EMPTY_IS_DATA:
            return ResolvedLineage(
                used_event_id=current_event_id, 
                freshness_state=FreshnessState.EMPTY_VALID, 
                stale_age_seconds=0, 
                payload_class=pclass, 
                na_reason=None
            )
            
        elif empty_policy == EmptyPayloadPolicy.EMPTY_MEANS_STALE:
            if prev_state and prev_state.get("last_success_event_id"):
                prev_success_ts = to_utc_dt(prev_state.get("last_success_ts_utc"), fallback=current_ts)
                age = max(0, int((current_ts - prev_success_ts).total_seconds()))
                return ResolvedLineage(
                    used_event_id=prev_state["last_success_event_id"], 
                    freshness_state=FreshnessState.STALE_CARRY, 
                    stale_age_seconds=age, 
                    payload_class=pclass, 
                    na_reason=f"CARRY_FORWARD_{pclass.name}"
                )
            else:
                return ResolvedLineage(
                    used_event_id=None, 
                    freshness_state=FreshnessState.EMPTY_VALID, 
                    stale_age_seconds=None, 
                    payload_class=pclass, 
                    na_reason="NO_PRIOR_SUCCESS"
                )

    if pclass == EndpointPayloadClass.ERROR:
        if prev_state and prev_state.get("last_success_event_id"):
            prev_success_ts = to_utc_dt(prev_state.get("last_success_ts_utc"), fallback=current_ts)
            age = max(0, int((current_ts - prev_success_ts).total_seconds()))
            return ResolvedLineage(
                used_event_id=prev_state["last_success_event_id"], 
                freshness_state=FreshnessState.STALE_CARRY, 
                stale_age_seconds=age, 
                payload_class=pclass, 
                na_reason="CARRY_FORWARD_ERROR"
            )
        else:
            return ResolvedLineage(
                used_event_id=None, 
                freshness_state=FreshnessState.ERROR, 
                stale_age_seconds=None, 
                payload_class=pclass, 
                na_reason="NO_PRIOR_SUCCESS"
            )
            
    return ResolvedLineage(None, FreshnessState.ERROR, None, pclass, "UNRESOLVED")