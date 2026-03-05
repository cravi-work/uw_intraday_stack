from __future__ import annotations
from enum import Enum
from dataclasses import dataclass, replace
from typing import Optional, List, Dict, Any, Tuple, Mapping, Sequence
from collections import deque
from email.utils import parsedate_to_datetime
import datetime
import math

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
    MISSING_PROVIDER_TIME = "MISSING_PROVIDER_TIME"
    TIME_PROVENANCE_DEGRADED = "TIME_PROVENANCE_DEGRADED"


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
    endpoint_asof_ts_utc: Optional[datetime.datetime] = None
    alignment_delta_sec: Optional[int] = None
    effective_ts_utc: Optional[datetime.datetime] = None
    event_time_utc: Optional[datetime.datetime] = None
    source_publish_time_utc: Optional[datetime.datetime] = None
    received_at_utc: Optional[datetime.datetime] = None
    processed_at_utc: Optional[datetime.datetime] = None
    as_of_time_utc: Optional[datetime.datetime] = None
    source_revision: Optional[str] = None
    effective_time_source: Optional[str] = None
    timestamp_quality: Optional[str] = None
    lagged: bool = False
    time_provenance_degraded: bool = False
    endpoint_name: Optional[str] = None
    endpoint_purpose: Optional[str] = None
    decision_path: Optional[bool] = None
    missing_affects_confidence: Optional[bool] = None
    stale_affects_confidence: Optional[bool] = None
    time_semantics_family: Optional[str] = None
    max_trusted_source_age_seconds: Optional[int] = None
    purpose_contract_version: Optional[str] = None


@dataclass(frozen=True)
class EndpointTimeSemantics:
    family: str = "default"
    documented_asof_contemporaneous: bool = False
    max_trusted_source_age_seconds: Optional[int] = None
    lagged: bool = False


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
    effective_ts_utc: Optional[datetime.datetime] = None
    event_time_utc: Optional[datetime.datetime] = None
    source_publish_time_utc: Optional[datetime.datetime] = None
    received_at_utc: Optional[datetime.datetime] = None
    processed_at_utc: Optional[datetime.datetime] = None
    as_of_time_utc: Optional[datetime.datetime] = None
    source_revision: Optional[str] = None
    effective_time_source: Optional[str] = None
    timestamp_quality: Optional[str] = None
    lagged: bool = False
    time_provenance_degraded: bool = False


@dataclass(frozen=True)
class SourceTimeHints:
    event_time_utc: Optional[datetime.datetime] = None
    source_publish_time_utc: Optional[datetime.datetime] = None
    effective_time_utc: Optional[datetime.datetime] = None
    source_revision: Optional[str] = None


PAYLOAD_EVENT_TIME_KEYS: Tuple[str, ...] = (
    "event_time", "event_at", "executed_at", "trade_time", "occurred_at",
    "timestamp", "time", "t", "ts", "date", "datetime", "date_time",
    "updated_at", "last_updated", "created_at", "start", "end",
    "eventTime", "executedAt", "tradeTime", "occurredAt", "lastUpdated",
    "createdAt", "dateTime", "quoteTime", "quote_time", "snapshotDate",
    "snapshot_date", "event_timestamp", "eventTimestamp", "trade_timestamp", "tradeTimestamp",
)
PAYLOAD_PUBLISH_TIME_KEYS: Tuple[str, ...] = (
    "source_publish_time", "source_publish_time_utc", "published_at",
    "publish_time", "report_time", "report_date", "generated_at", "created_at",
    "publishedAt", "publishTime", "reportTime", "generatedAt", "createdAt",
    "generated_time", "generatedTime", "publication_time", "publicationTime",
)
PAYLOAD_EFFECTIVE_TIME_KEYS: Tuple[str, ...] = (
    "effective_at", "effective_ts", "effective_ts_utc", "effective_time", "as_of",
    "effectiveAt", "effectiveTs", "effectiveTime", "as_of_time", "asOf",
    "asOfTime", "snapshot_time", "snapshotTime", "snapshot_at", "snapshotAt",
    "snapshot_date", "snapshotDate", "current_date", "currentDate",
)
PAYLOAD_SOURCE_REVISION_KEYS: Tuple[str, ...] = (
    "source_revision", "revision", "rev", "version", "sequence_id", "update_id",
    "sourceRevision", "sequenceId", "updateId",
)
RESPONSE_HEADER_EVENT_KEYS: Tuple[str, ...] = (
    "x-source-event-time", "x-event-time", "x-provider-event-time",
)
RESPONSE_HEADER_PUBLISH_KEYS: Tuple[str, ...] = (
    "x-source-publish-time", "x-published-at", "last-modified",
    "x-generated-at", "x-response-generated-at", "x-provider-publish-time",
)
RESPONSE_HEADER_EFFECTIVE_KEYS: Tuple[str, ...] = (
    "x-effective-time", "x-as-of-time", "x-data-as-of", "x-snapshot-time",
)
RESPONSE_HEADER_REVISION_KEYS: Tuple[str, ...] = (
    "x-source-revision", "x-revision", "etag", "x-version", "x-data-revision",
)

DOCUMENTED_ASOF_CONTEMPORANEOUS_PATHS: Tuple[str, ...] = (
    "/api/stock/{ticker}/historical-risk-reversal-skew",
    "/api/stock/{ticker}/spot-exposures",
    "/api/stock/{ticker}/spot-exposures/strike",
    "/api/stock/{ticker}/spot-exposures/expiry-strike",
    "/api/stock/{ticker}/greek-exposure",
    "/api/stock/{ticker}/greek-exposure/strike",
    "/api/stock/{ticker}/greek-exposure/expiry",
    "/api/stock/{ticker}/oi-per-strike",
    "/api/stock/{ticker}/oi-change",
    "/api/stock/{ticker}/iv-rank",
    "/api/stock/{ticker}/volatility/term-structure",
)


ENDPOINT_TIME_SEMANTICS: Dict[str, EndpointTimeSemantics] = {
    "/api/stock/{ticker}/spot-exposures": EndpointTimeSemantics(
        family="gex_snapshot",
        documented_asof_contemporaneous=True,
        max_trusted_source_age_seconds=7200,
        lagged=True,
    ),
    "/api/stock/{ticker}/spot-exposures/strike": EndpointTimeSemantics(
        family="gex_snapshot",
        documented_asof_contemporaneous=True,
        max_trusted_source_age_seconds=7200,
        lagged=True,
    ),
    "/api/stock/{ticker}/spot-exposures/expiry-strike": EndpointTimeSemantics(
        family="gex_snapshot",
        documented_asof_contemporaneous=True,
        max_trusted_source_age_seconds=7200,
        lagged=True,
    ),
    "/api/stock/{ticker}/greek-exposure": EndpointTimeSemantics(
        family="greeks_snapshot",
        documented_asof_contemporaneous=True,
        max_trusted_source_age_seconds=7200,
        lagged=True,
    ),
    "/api/stock/{ticker}/greek-exposure/strike": EndpointTimeSemantics(
        family="greeks_snapshot",
        documented_asof_contemporaneous=True,
        max_trusted_source_age_seconds=7200,
        lagged=True,
    ),
    "/api/stock/{ticker}/greek-exposure/expiry": EndpointTimeSemantics(
        family="greeks_snapshot",
        documented_asof_contemporaneous=True,
        max_trusted_source_age_seconds=7200,
        lagged=True,
    ),
    "/api/stock/{ticker}/oi-per-strike": EndpointTimeSemantics(
        family="oi_snapshot",
        documented_asof_contemporaneous=True,
        max_trusted_source_age_seconds=7200,
        lagged=True,
    ),
    "/api/stock/{ticker}/oi-change": EndpointTimeSemantics(
        family="oi_snapshot",
        documented_asof_contemporaneous=True,
        max_trusted_source_age_seconds=7200,
        lagged=True,
    ),
    "/api/stock/{ticker}/iv-rank": EndpointTimeSemantics(
        family="vol_snapshot",
        documented_asof_contemporaneous=True,
        max_trusted_source_age_seconds=7200,
        lagged=True,
    ),
    "/api/stock/{ticker}/volatility/term-structure": EndpointTimeSemantics(
        family="vol_snapshot",
        documented_asof_contemporaneous=True,
        max_trusted_source_age_seconds=7200,
        lagged=True,
    ),
    "/api/stock/{ticker}/historical-risk-reversal-skew": EndpointTimeSemantics(
        family="vol_snapshot",
        documented_asof_contemporaneous=True,
        max_trusted_source_age_seconds=7200,
        lagged=True,
    ),
}


def get_endpoint_time_semantics(path: Optional[str]) -> EndpointTimeSemantics:
    if not path:
        return EndpointTimeSemantics()
    return ENDPOINT_TIME_SEMANTICS.get(str(path), EndpointTimeSemantics())


def uses_documented_asof_contemporaneous(path: Optional[str]) -> bool:
    return bool(get_endpoint_time_semantics(path).documented_asof_contemporaneous)


def _normalize_contract_key(key: Any) -> str:
    return "".join(ch for ch in str(key).lower() if ch.isalnum())


def _normalize_epoch_seconds(value: float) -> Optional[float]:
    if not math.isfinite(float(value)):
        return None
    epoch_value = float(value)
    magnitude = abs(epoch_value)
    if magnitude >= 1e17:
        epoch_value /= 1_000_000_000.0
    elif magnitude >= 1e14:
        epoch_value /= 1_000_000.0
    elif magnitude >= 1e11:
        epoch_value /= 1_000.0
    return epoch_value


def _coerce_optional_utc_dt(x: Any) -> Optional[datetime.datetime]:
    if x is None:
        return None
    if isinstance(x, datetime.datetime):
        if x.tzinfo is None:
            return x.replace(tzinfo=datetime.timezone.utc)
        return x.astimezone(datetime.timezone.utc)
    if isinstance(x, (int, float)):
        epoch_value = _normalize_epoch_seconds(float(x))
        if epoch_value is None:
            return None
        try:
            return datetime.datetime.fromtimestamp(epoch_value, datetime.timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    if isinstance(x, str):
        raw = x.strip()
        if not raw:
            return None
        try:
            parsed = datetime.datetime.fromisoformat(raw.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=datetime.timezone.utc)
            else:
                parsed = parsed.astimezone(datetime.timezone.utc)
            return parsed
        except ValueError:
            try:
                parsed = parsedate_to_datetime(raw)
                if parsed is not None:
                    if parsed.tzinfo is None:
                        parsed = parsed.replace(tzinfo=datetime.timezone.utc)
                    return parsed.astimezone(datetime.timezone.utc)
            except (TypeError, ValueError, OverflowError, IndexError):
                pass
            try:
                epoch_value = _normalize_epoch_seconds(float(raw))
                if epoch_value is None:
                    return None
                return datetime.datetime.fromtimestamp(epoch_value, datetime.timezone.utc)
            except (TypeError, ValueError, OverflowError, OSError):
                return None
    return None


def _find_best_nested_timestamp(payload: Any, candidate_keys: Sequence[str], *, max_nodes: int = 256) -> Optional[datetime.datetime]:
    if payload is None:
        return None
    keyset = {_normalize_contract_key(k) for k in candidate_keys}
    queue = deque([payload])
    seen = 0
    best_dt: Optional[datetime.datetime] = None

    while queue and seen < max_nodes:
        current = queue.popleft()
        seen += 1
        if isinstance(current, Mapping):
            for key, value in current.items():
                if _normalize_contract_key(key) in keyset:
                    dt_val = _coerce_optional_utc_dt(value)
                    if dt_val is not None and (best_dt is None or dt_val > best_dt):
                        best_dt = dt_val
            for value in current.values():
                if isinstance(value, (Mapping, list, tuple)):
                    queue.append(value)
        elif isinstance(current, (list, tuple)):
            for value in current[:32]:
                if isinstance(value, (Mapping, list, tuple)):
                    queue.append(value)
    return best_dt


def _find_first_nested_value(payload: Any, candidate_keys: Sequence[str], *, max_nodes: int = 256) -> Any:
    if payload is None:
        return None
    keyset = {_normalize_contract_key(k) for k in candidate_keys}
    queue = deque([payload])
    seen = 0

    while queue and seen < max_nodes:
        current = queue.popleft()
        seen += 1
        if isinstance(current, Mapping):
            for key, value in current.items():
                if _normalize_contract_key(key) in keyset and value not in (None, ""):
                    return value
            for value in current.values():
                if isinstance(value, (Mapping, list, tuple)):
                    queue.append(value)
        elif isinstance(current, (list, tuple)):
            for value in current[:32]:
                if isinstance(value, (Mapping, list, tuple)):
                    queue.append(value)
    return None


def _normalize_response_headers(headers: Any) -> Dict[str, str]:
    if not isinstance(headers, Mapping):
        return {}
    out: Dict[str, str] = {}
    for key, value in headers.items():
        if value in (None, ""):
            continue
        out[str(key).strip().lower()] = str(value).strip()
    return out


def infer_source_time_hints(
    *,
    payload_json: Any = None,
    response_headers: Any = None,
    explicit_event_time_raw: Any = None,
    explicit_publish_time_raw: Any = None,
    explicit_effective_time_raw: Any = None,
    explicit_revision: Optional[str] = None,
) -> SourceTimeHints:
    headers = _normalize_response_headers(response_headers)
    normalized_headers = {
        _normalize_contract_key(key): value
        for key, value in headers.items()
        if value not in (None, "")
    }

    explicit_event = _coerce_optional_utc_dt(explicit_event_time_raw)
    explicit_publish = _coerce_optional_utc_dt(explicit_publish_time_raw)
    explicit_effective = _coerce_optional_utc_dt(explicit_effective_time_raw)

    payload_event = _find_best_nested_timestamp(payload_json, PAYLOAD_EVENT_TIME_KEYS)
    payload_publish = _find_best_nested_timestamp(payload_json, PAYLOAD_PUBLISH_TIME_KEYS)
    payload_effective = _find_best_nested_timestamp(payload_json, PAYLOAD_EFFECTIVE_TIME_KEYS)
    payload_revision = _find_first_nested_value(payload_json, PAYLOAD_SOURCE_REVISION_KEYS)

    header_event = None
    for key in RESPONSE_HEADER_EVENT_KEYS:
        header_event = _coerce_optional_utc_dt(normalized_headers.get(_normalize_contract_key(key)))
        if header_event is not None:
            break

    header_publish = None
    for key in RESPONSE_HEADER_PUBLISH_KEYS:
        header_publish = _coerce_optional_utc_dt(normalized_headers.get(_normalize_contract_key(key)))
        if header_publish is not None:
            break

    header_effective = None
    for key in RESPONSE_HEADER_EFFECTIVE_KEYS:
        header_effective = _coerce_optional_utc_dt(normalized_headers.get(_normalize_contract_key(key)))
        if header_effective is not None:
            break

    header_revision = None
    for key in RESPONSE_HEADER_REVISION_KEYS:
        header_value = normalized_headers.get(_normalize_contract_key(key))
        if header_value not in (None, ""):
            header_revision = header_value
            break

    return SourceTimeHints(
        event_time_utc=explicit_event or payload_event or header_event,
        source_publish_time_utc=explicit_publish or payload_publish or header_publish,
        effective_time_utc=explicit_effective or payload_effective or header_effective,
        source_revision=str(explicit_revision) if explicit_revision not in (None, "") else (str(payload_revision) if payload_revision not in (None, "") else header_revision),
    )


def to_utc_dt(x: Any, *, fallback: datetime.datetime) -> datetime.datetime:
    coerced = _coerce_optional_utc_dt(x)
    return coerced if coerced is not None else fallback


def _merge_reason(existing: Optional[str], new_reason: Optional[str]) -> Optional[str]:
    if not new_reason:
        return existing
    if not existing:
        return new_reason
    parts = [p for p in str(existing).split("|") if p]
    if new_reason in parts:
        return existing
    parts.append(new_reason)
    return "|".join(parts)


def _resolve_source_time(
    *,
    source_event_time_raw: Any,
    source_publish_time_raw: Any,
    effective_time_raw: Any,
    as_of_time: datetime.datetime,
    documented_asof_contemporaneous: bool,
    max_trusted_source_age_seconds: Optional[int] = None,
    time_semantics_family: Optional[str] = None,
) -> Tuple[Optional[datetime.datetime], Optional[datetime.datetime], Optional[datetime.datetime], Optional[str], str, bool, Optional[str]]:
    event_time = _coerce_optional_utc_dt(source_event_time_raw)
    source_publish_time = _coerce_optional_utc_dt(source_publish_time_raw)
    explicit_effective_time = _coerce_optional_utc_dt(effective_time_raw)

    def _reclassify_out_of_window(
        candidate_ts: Optional[datetime.datetime],
        *,
        source_name: str,
    ) -> Optional[Tuple[Optional[datetime.datetime], Optional[datetime.datetime], Optional[datetime.datetime], Optional[str], str, bool, Optional[str]]]:
        if candidate_ts is None or not documented_asof_contemporaneous:
            return None
        if max_trusted_source_age_seconds is None:
            return None
        age_seconds = int((as_of_time - candidate_ts).total_seconds())
        if age_seconds <= max_trusted_source_age_seconds:
            return None
        family = str(time_semantics_family or "snapshot")
        reason = _merge_reason(
            NaReasonCode.TIME_PROVENANCE_DEGRADED.value,
            f"{family}_provider_time_out_of_window:{age_seconds}s_gt_{max_trusted_source_age_seconds}s",
        )
        return (
            as_of_time,
            event_time,
            source_publish_time,
            "documented_asof_contemporaneous",
            "DEGRADED",
            True,
            reason,
        )

    if explicit_effective_time is not None:
        reclassified = _reclassify_out_of_window(explicit_effective_time, source_name="payload_effective_time")
        if reclassified is not None:
            return reclassified
        return (
            explicit_effective_time,
            event_time,
            source_publish_time,
            "payload_effective_time",
            "VALID",
            False,
            None,
        )

    if event_time is not None:
        reclassified = _reclassify_out_of_window(event_time, source_name="event_time")
        if reclassified is not None:
            return reclassified
        return (
            event_time,
            event_time,
            source_publish_time,
            "event_time",
            "VALID",
            False,
            None,
        )

    if source_publish_time is not None:
        reclassified = _reclassify_out_of_window(source_publish_time, source_name="source_publish_time")
        if reclassified is not None:
            return reclassified
        return (
            source_publish_time,
            event_time,
            source_publish_time,
            "source_publish_time",
            "VALID",
            False,
            None,
        )

    if documented_asof_contemporaneous:
        reason = _merge_reason(NaReasonCode.TIME_PROVENANCE_DEGRADED.value, NaReasonCode.MISSING_PROVIDER_TIME.value)
        return (
            as_of_time,
            None,
            None,
            "documented_asof_contemporaneous",
            "DEGRADED",
            True,
            reason,
        )

    reason = _merge_reason(NaReasonCode.TIME_PROVENANCE_DEGRADED.value, NaReasonCode.MISSING_PROVIDER_TIME.value)
    return (None, None, None, "missing_provider_time", "DEGRADED", True, reason)


def _build_resolved_lineage(
    *,
    used_event_id: Optional[str],
    freshness_state: FreshnessState,
    stale_age_seconds: Optional[int],
    payload_class: EndpointPayloadClass,
    na_reason: Optional[str],
    effective_ts_utc: Optional[datetime.datetime],
    event_time_utc: Optional[datetime.datetime],
    source_publish_time_utc: Optional[datetime.datetime],
    received_at_utc: Optional[datetime.datetime],
    processed_at_utc: Optional[datetime.datetime],
    as_of_time_utc: Optional[datetime.datetime],
    source_revision: Optional[str],
    effective_time_source: Optional[str],
    timestamp_quality: Optional[str],
    lagged: bool,
    time_provenance_degraded: bool,
) -> ResolvedLineage:
    return ResolvedLineage(
        used_event_id=used_event_id,
        freshness_state=freshness_state,
        stale_age_seconds=stale_age_seconds,
        payload_class=payload_class,
        na_reason=na_reason,
        effective_ts_utc=effective_ts_utc,
        event_time_utc=event_time_utc,
        source_publish_time_utc=source_publish_time_utc,
        received_at_utc=received_at_utc,
        processed_at_utc=processed_at_utc,
        as_of_time_utc=as_of_time_utc,
        source_revision=source_revision,
        effective_time_source=effective_time_source,
        timestamp_quality=timestamp_quality,
        lagged=lagged,
        time_provenance_degraded=time_provenance_degraded,
    )


def is_provider_error_envelope(pj: Any) -> bool:
    if not isinstance(pj, dict):
        return False
    keys = set(pj.keys())
    error_keys = {"error", "message", "detail", "status"}
    data_containers = {"data", "results", "items", "trades", "history"}
    return bool(keys.intersection(error_keys) and not keys.intersection(data_containers))


def validate_shape(pj: Any, rule: Optional[EndpointRule]) -> Tuple[bool, Optional[str], Optional[List[str]]]:
    if not rule or not isinstance(pj, dict):
        return True, None, None
    actual_pj = pj
    for container in rule.data_container_keys:
        if container in pj and len(pj.keys()) <= 2:
            actual_pj = pj[container]
            break
    if not isinstance(actual_pj, dict):
        return True, None, None

    keys = set(actual_pj.keys())
    if rule.required_all_keys:
        missing = [k for k in rule.required_all_keys if k not in keys]
        if missing:
            return False, "INVALID_SHAPE:Missing required_all_keys", missing
    if rule.required_any_keys:
        if not any(k in keys for k in rule.required_any_keys):
            return False, "INVALID_SHAPE:Missing required_any_keys", list(rule.required_any_keys)
    return True, None, None


def _enforce_invariants(res: ResolvedLineage) -> ResolvedLineage:
    if res.freshness_state == FreshnessState.STALE_CARRY and not res.used_event_id:
        return replace(
            res,
            used_event_id=None,
            freshness_state=FreshnessState.ERROR,
            stale_age_seconds=None,
            na_reason=NaReasonCode.NO_PRIOR_SUCCESS.value,
        )
    return res


def _apply_stale_cutoff(res: ResolvedLineage, age: int, fallback_max_age: int, invalid_after: int) -> ResolvedLineage:
    if age > invalid_after:
        return replace(res, freshness_state=FreshnessState.ERROR, used_event_id=None, na_reason=NaReasonCode.STALE_TOO_OLD.value)
    if age > fallback_max_age:
        warn_str = NaReasonCode.STALE_WARN.value
        new_reason = _merge_reason(res.na_reason, warn_str)
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
    if isinstance(pj, dict) and not pj:
        is_empty = True
    if isinstance(pj, list) and not pj:
        is_empty = True
    if isinstance(pj, dict) and len(pj) == 1:
        if "data" in pj and not pj["data"]:
            is_empty = True
        if "results" in pj and not pj["results"]:
            is_empty = True

    computed_changed = True if prev_hash is None else (res.payload_hash != prev_hash)

    if is_empty:
        if policy == EmptyPayloadPolicy.EMPTY_IS_DATA:
            return PayloadAssessment(EndpointPayloadClass.SUCCESS_EMPTY_VALID, policy, True, computed_changed, None)
        if policy == EmptyPayloadPolicy.EMPTY_MEANS_STALE:
            return PayloadAssessment(EndpointPayloadClass.SUCCESS_EMPTY_VALID, policy, True, False, "EMPTY_MEANS_STALE")
        return PayloadAssessment(EndpointPayloadClass.ERROR, policy, True, None, "EMPTY_UNEXPECTED")

    is_valid, shape_err, missing = validate_shape(pj, rule)
    if not is_valid:
        return PayloadAssessment(EndpointPayloadClass.ERROR, policy, False, None, shape_err, missing, "SHAPE_VALIDATOR")

    pclass = EndpointPayloadClass.SUCCESS_HAS_DATA if computed_changed else EndpointPayloadClass.SUCCESS_STALE
    return PayloadAssessment(pclass, policy, False, computed_changed, None)


def _get_carry_forward_target(prev_state: Optional[EndpointStateRow], current_ts: datetime.datetime) -> Tuple[Optional[str], Optional[int]]:
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
    current_event_id: str,
    current_ts_raw: Any,
    assessment: PayloadAssessment,
    prev_state: Optional[EndpointStateRow],
    fallback_max_age_seconds: int = 900,
    invalid_after_seconds: int = 3600,
    *,
    source_event_time_raw: Any = None,
    source_publish_time_raw: Any = None,
    received_at_raw: Any = None,
    processed_at_raw: Any = None,
    effective_time_raw: Any = None,
    as_of_time_raw: Any = None,
    source_revision: Optional[str] = None,
    documented_asof_contemporaneous: bool = False,
    max_trusted_source_age_seconds: Optional[int] = None,
    time_semantics_family: Optional[str] = None,
) -> ResolvedLineage:
    current_ts = to_utc_dt(current_ts_raw, fallback=datetime.datetime.now(datetime.timezone.utc))
    as_of_time = to_utc_dt(as_of_time_raw, fallback=current_ts)
    received_at = _coerce_optional_utc_dt(received_at_raw)
    processed_at = to_utc_dt(processed_at_raw, fallback=current_ts)
    pclass = assessment.payload_class
    empty_policy = assessment.empty_policy

    if pclass == EndpointPayloadClass.SUCCESS_HAS_DATA:
        effective_ts, event_time, source_publish_time, source_name, ts_quality, degraded, reason = _resolve_source_time(
            source_event_time_raw=source_event_time_raw,
            source_publish_time_raw=source_publish_time_raw,
            effective_time_raw=effective_time_raw,
            as_of_time=as_of_time,
            documented_asof_contemporaneous=documented_asof_contemporaneous,
            max_trusted_source_age_seconds=max_trusted_source_age_seconds,
            time_semantics_family=time_semantics_family,
        )
        resolved = _build_resolved_lineage(
            used_event_id=current_event_id,
            freshness_state=FreshnessState.FRESH,
            stale_age_seconds=0,
            payload_class=pclass,
            na_reason=reason,
            effective_ts_utc=effective_ts,
            event_time_utc=event_time,
            source_publish_time_utc=source_publish_time,
            received_at_utc=received_at,
            processed_at_utc=processed_at,
            as_of_time_utc=as_of_time,
            source_revision=source_revision,
            effective_time_source=source_name,
            timestamp_quality=ts_quality,
            lagged=bool(source_name == "documented_asof_contemporaneous"),
            time_provenance_degraded=degraded,
        )
        return _enforce_invariants(resolved)

    if pclass == EndpointPayloadClass.SUCCESS_STALE:
        used_id = prev_state.last_change_event_id if prev_state and prev_state.last_change_event_id else current_event_id
        prev_change_ts = to_utc_dt(prev_state.last_change_ts_utc, fallback=current_ts) if prev_state and prev_state.last_change_ts_utc else current_ts
        age = max(0, int((current_ts - prev_change_ts).total_seconds()))
        eff_ts = to_utc_dt(prev_state.last_success_ts_utc, fallback=current_ts) if prev_state and prev_state.last_success_ts_utc else current_ts
        resolved = _build_resolved_lineage(
            used_event_id=used_id,
            freshness_state=FreshnessState.STALE_CARRY,
            stale_age_seconds=age,
            payload_class=pclass,
            na_reason=None,
            effective_ts_utc=eff_ts,
            event_time_utc=eff_ts,
            source_publish_time_utc=None,
            received_at_utc=received_at,
            processed_at_utc=processed_at,
            as_of_time_utc=as_of_time,
            source_revision=source_revision,
            effective_time_source="carry_forward_last_success",
            timestamp_quality="LAGGED",
            lagged=True,
            time_provenance_degraded=False,
        )
        return _enforce_invariants(_apply_stale_cutoff(resolved, age, fallback_max_age_seconds, invalid_after_seconds))

    if pclass == EndpointPayloadClass.SUCCESS_EMPTY_VALID:
        effective_ts, event_time, source_publish_time, source_name, ts_quality, degraded, reason = _resolve_source_time(
            source_event_time_raw=source_event_time_raw,
            source_publish_time_raw=source_publish_time_raw,
            effective_time_raw=effective_time_raw,
            as_of_time=as_of_time,
            documented_asof_contemporaneous=documented_asof_contemporaneous,
        )
        if empty_policy == EmptyPayloadPolicy.EMPTY_IS_DATA:
            resolved = _build_resolved_lineage(
                used_event_id=current_event_id,
                freshness_state=FreshnessState.EMPTY_VALID,
                stale_age_seconds=0,
                payload_class=pclass,
                na_reason=reason,
                effective_ts_utc=effective_ts,
                event_time_utc=event_time,
                source_publish_time_utc=source_publish_time,
                received_at_utc=received_at,
                processed_at_utc=processed_at,
                as_of_time_utc=as_of_time,
                source_revision=source_revision,
                effective_time_source=source_name,
                timestamp_quality=ts_quality,
                lagged=False,
                time_provenance_degraded=degraded,
            )
            return _enforce_invariants(resolved)
        if empty_policy == EmptyPayloadPolicy.EMPTY_MEANS_STALE:
            cf_id, age = _get_carry_forward_target(prev_state, current_ts)
            if cf_id:
                eff_ts = to_utc_dt(prev_state.last_success_ts_utc, fallback=current_ts) if prev_state and prev_state.last_success_ts_utc else current_ts
                resolved = _build_resolved_lineage(
                    used_event_id=cf_id,
                    freshness_state=FreshnessState.STALE_CARRY,
                    stale_age_seconds=age,
                    payload_class=pclass,
                    na_reason=NaReasonCode.CARRY_FORWARD_EMPTY_MEANS_STALE.value,
                    effective_ts_utc=eff_ts,
                    event_time_utc=eff_ts,
                    source_publish_time_utc=None,
                    received_at_utc=received_at,
                    processed_at_utc=processed_at,
                    as_of_time_utc=as_of_time,
                    source_revision=source_revision,
                    effective_time_source="carry_forward_last_success",
                    timestamp_quality="LAGGED",
                    lagged=True,
                    time_provenance_degraded=False,
                )
                return _enforce_invariants(_apply_stale_cutoff(resolved, age, fallback_max_age_seconds, invalid_after_seconds))
            return _enforce_invariants(
                _build_resolved_lineage(
                    used_event_id=None,
                    freshness_state=FreshnessState.ERROR,
                    stale_age_seconds=None,
                    payload_class=pclass,
                    na_reason=NaReasonCode.NO_PRIOR_SUCCESS.value,
                    effective_ts_utc=None,
                    event_time_utc=None,
                    source_publish_time_utc=None,
                    received_at_utc=received_at,
                    processed_at_utc=processed_at,
                    as_of_time_utc=as_of_time,
                    source_revision=source_revision,
                    effective_time_source="missing_provider_time",
                    timestamp_quality="INVALID",
                    lagged=False,
                    time_provenance_degraded=True,
                )
            )

    if pclass == EndpointPayloadClass.ERROR:
        cf_id, age = _get_carry_forward_target(prev_state, current_ts)
        if cf_id:
            eff_ts = to_utc_dt(prev_state.last_success_ts_utc, fallback=current_ts) if prev_state and prev_state.last_success_ts_utc else current_ts
            resolved = _build_resolved_lineage(
                used_event_id=cf_id,
                freshness_state=FreshnessState.STALE_CARRY,
                stale_age_seconds=age,
                payload_class=pclass,
                na_reason=NaReasonCode.CARRY_FORWARD_ERROR.value,
                effective_ts_utc=eff_ts,
                event_time_utc=eff_ts,
                source_publish_time_utc=None,
                received_at_utc=received_at,
                processed_at_utc=processed_at,
                as_of_time_utc=as_of_time,
                source_revision=source_revision,
                effective_time_source="carry_forward_last_success",
                timestamp_quality="LAGGED",
                lagged=True,
                time_provenance_degraded=False,
            )
            return _enforce_invariants(_apply_stale_cutoff(resolved, age, fallback_max_age_seconds, invalid_after_seconds))
        return _enforce_invariants(
            _build_resolved_lineage(
                used_event_id=None,
                freshness_state=FreshnessState.ERROR,
                stale_age_seconds=None,
                payload_class=pclass,
                na_reason=NaReasonCode.NO_PRIOR_SUCCESS.value,
                effective_ts_utc=None,
                event_time_utc=None,
                source_publish_time_utc=None,
                received_at_utc=received_at,
                processed_at_utc=processed_at,
                as_of_time_utc=as_of_time,
                source_revision=source_revision,
                effective_time_source="missing_provider_time",
                timestamp_quality="INVALID",
                lagged=False,
                time_provenance_degraded=True,
            )
        )

    return _enforce_invariants(
        _build_resolved_lineage(
            used_event_id=None,
            freshness_state=FreshnessState.ERROR,
            stale_age_seconds=None,
            payload_class=pclass,
            na_reason=NaReasonCode.UNRESOLVED.value,
            effective_ts_utc=None,
            event_time_utc=None,
            source_publish_time_utc=None,
            received_at_utc=received_at,
            processed_at_utc=processed_at,
            as_of_time_utc=as_of_time,
            source_revision=source_revision,
            effective_time_source="unresolved",
            timestamp_quality="INVALID",
            lagged=False,
            time_provenance_degraded=True,
        )
    )
