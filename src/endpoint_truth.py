from __future__ import annotations
from enum import Enum
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
import datetime

from .endpoint_rules import is_empty_valid

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

def _safe_utc(ts: Any) -> Optional[datetime.datetime]:
    """Safely coerces naive DuckDB timestamps, floats, or aware datetimes into standard UTC."""
    if not ts:
        return None
    if isinstance(ts, (int, float)):
        return datetime.datetime.fromtimestamp(ts, datetime.timezone.utc)
    if isinstance(ts, datetime.datetime):
        if ts.tzinfo is None:
            return ts.replace(tzinfo=datetime.timezone.utc)
        return ts.astimezone(datetime.timezone.utc)
    return None

def classify_payload(res: Any, prev_hash: Optional[str], method: str, path: str, session_label: str) -> Tuple[EndpointPayloadClass, Optional[str]]:
    if not getattr(res, "ok", False) or getattr(res, "payload_json", None) is None:
        return EndpointPayloadClass.ERROR, getattr(res, "error_message", "HTTP Error or Missing JSON")

    pj = res.payload_json
    is_empty = False
    if isinstance(pj, dict) and not pj: is_empty = True
    if isinstance(pj, list) and not pj: is_empty = True
    # Unwrap UW specific {"data": []} empty envelopes
    if isinstance(pj, dict) and len(pj) == 1 and "data" in pj and not pj["data"]: is_empty = True

    if is_empty:
        if is_empty_valid(method, path, session_label):
            return EndpointPayloadClass.SUCCESS_EMPTY_VALID, None
        return EndpointPayloadClass.ERROR, "EMPTY_UNEXPECTED"

    changed = (res.payload_hash != prev_hash)
    return EndpointPayloadClass.SUCCESS_HAS_DATA if changed else EndpointPayloadClass.SUCCESS_STALE, None

def resolve_effective_payload(
    current_event_id: str, 
    current_ts_raw: Any, 
    payload_class: EndpointPayloadClass, 
    prev_state: Optional[Dict[str, Any]]
) -> ResolvedLineage:
    
    current_ts = _safe_utc(current_ts_raw) or datetime.datetime.now(datetime.timezone.utc)

    if payload_class == EndpointPayloadClass.SUCCESS_HAS_DATA:
        return ResolvedLineage(current_event_id, FreshnessState.FRESH, 0, payload_class, None)
    
    if payload_class == EndpointPayloadClass.SUCCESS_STALE:
        used_id = prev_state["last_change_event_id"] if prev_state and prev_state.get("last_change_event_id") else current_event_id
        prev_change_ts = _safe_utc(prev_state.get("last_change_ts_utc")) if prev_state else None
        age = int((current_ts - prev_change_ts).total_seconds()) if prev_change_ts else 0
        return ResolvedLineage(used_id, FreshnessState.STALE_CARRY, max(0, age), payload_class, None)

    # Empty Valid or Error conditions -> Attempt carry-forward
    if prev_state and prev_state.get("last_success_event_id"):
        prev_success_ts = _safe_utc(prev_state.get("last_success_ts_utc"))
        age = int((current_ts - prev_success_ts).total_seconds()) if prev_success_ts else 0
        return ResolvedLineage(
            prev_state["last_success_event_id"], 
            FreshnessState.STALE_CARRY, 
            max(0, age), 
            payload_class, 
            f"CARRY_FORWARD_{payload_class.name}"
        )
    
    # Complete failure (no prior success to fall back on)
    freshness = FreshnessState.EMPTY_VALID if payload_class == EndpointPayloadClass.SUCCESS_EMPTY_VALID else FreshnessState.ERROR
    return ResolvedLineage(None, freshness, None, payload_class, "NO_PRIOR_SUCCESS")