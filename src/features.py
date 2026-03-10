# src/features.py
from __future__ import annotations
import copy
import datetime
import logging
import math
import re
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Mapping, Iterable
from dataclasses import dataclass

from .na import safe_float, is_na, grab_list
from .endpoint_truth import EndpointContext
from .analytics import (
    build_darkpool_levels,
    build_gex_levels,
    build_oi_walls,
    derived_level_usage_contract,
)
from .instruments import contract_scale, normalize_option_rows, normalized_contract_map
from .logging_config import structured_log

logger = logging.getLogger(__name__)

# Some UW endpoints only provide a daily "date" (no timestamp). For those snapshot-style
# metrics we allow a deterministic default to midnight UTC of the provided date.
_DATE_ONLY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

_grab_list = grab_list 
_as_float = safe_float 


def _parse_utc_dt(value: Any) -> Optional[datetime.datetime]:
    """Best-effort coercion of payload time values to timezone-aware UTC datetimes."""
    if value is None:
        return None
    if isinstance(value, datetime.datetime):
        dt = value
    elif isinstance(value, datetime.date):
        dt = datetime.datetime(value.year, value.month, value.day)
    elif isinstance(value, (int, float)):
        # Treat as epoch seconds.
        try:
            dt = datetime.datetime.fromtimestamp(float(value), tz=datetime.timezone.utc)
        except Exception:
            return None
    elif isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        # Accept trailing Z.
        s = s.replace("Z", "+00:00")
        try:
            dt = datetime.datetime.fromisoformat(s)
        except Exception:
            return None
    else:
        return None

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    else:
        dt = dt.astimezone(datetime.timezone.utc)
    return dt


def _infer_daily_snapshot_effective_ts_utc(payload: Any) -> Optional[datetime.datetime]:
    """Infer an effective timestamp for snapshot endpoints that only return a date.

    Heuristic (intentionally narrow and deterministic):
    - Prefer a top-level "date" when present.
    - Otherwise look at the first row in a common container.
    - Only apply when the value is date-only (YYYY-MM-DD), and default to midnight UTC.
    """

    def _coerce_date_only(val: Any) -> Optional[datetime.datetime]:
        if not isinstance(val, str):
            return None
        s = val.strip()
        if not _DATE_ONLY_RE.match(s):
            return None
        dt = _parse_utc_dt(s)
        if dt is None:
            return None
        return dt.replace(hour=0, minute=0, second=0, microsecond=0)

    if isinstance(payload, dict):
        # 1) Top-level date
        top = _coerce_date_only(payload.get("date"))
        if top is not None:
            return top
        for k in ("as_of", "asof", "snapshot_date", "snapshotDate"):
            top = _coerce_date_only(payload.get(k))
            if top is not None:
                return top

        # 2) Date on rows in common containers
        for container_key in ("data", "results", "result", "items"):
            rows = payload.get(container_key)
            if isinstance(rows, list) and rows:
                # Some endpoints return newest-last; others return newest-first.
                # Check a small, deterministic sample: first, last, then up to 3 more.
                sample_rows = []
                sample_rows.append(rows[0])
                if len(rows) > 1:
                    sample_rows.append(rows[-1])
                    sample_rows.extend(rows[1:4])

                best_row_date: Optional[dt.datetime] = None
                for row in sample_rows:
                    if not isinstance(row, dict):
                        continue
                    for k in ("date", "as_of", "asof", "snapshot_date", "snapshotDate", "asOf"):
                        row_date = _coerce_date_only(row.get(k))
                        if row_date is not None:
                            if best_row_date is None or row_date > best_row_date:
                                best_row_date = row_date

                if best_row_date is not None:
                    return best_row_date

    elif isinstance(payload, list) and payload:
        sample_rows = [payload[0]]
        if len(payload) > 1:
            sample_rows.append(payload[-1])
            sample_rows.extend(payload[1:4])

        best_row_date: Optional[dt.datetime] = None
        for row in sample_rows:
            if not isinstance(row, dict):
                continue
            for k in ("date", "as_of", "asof", "snapshot_date", "snapshotDate", "asOf"):
                row_date = _coerce_date_only(row.get(k))
                if row_date is not None:
                    if best_row_date is None or row_date > best_row_date:
                        best_row_date = row_date

        if best_row_date is not None:
            return best_row_date
    return None

class FeatureRow(TypedDict):
    feature_key: str
    feature_value: Optional[float]
    meta_json: Dict[str, Any]

class LevelRow(TypedDict):
    level_type: str
    price: Optional[float]
    magnitude: Optional[float]
    meta_json: Dict[str, Any]

@dataclass
class FeatureBundle:
    features: Dict[str, Optional[float]]
    meta: Dict[str, Any]

@dataclass
class FeatureCandidate:
    feature_key: str
    feature_value: Optional[float]
    meta_json: Dict[str, Any]
    freshness_rank: int
    stale_age: int
    path_priority: int
    endpoint_id: int
    is_none: bool

PATH_PRIORITY = {
    "/api/stock/{ticker}/spot-exposures": 1,
    "/api/stock/{ticker}/spot-exposures/strike": 2,
    "/api/stock/{ticker}/spot-exposures/expiry-strike": 3,
    "/api/stock/{ticker}/flow-recent": 1,
    "/api/stock/{ticker}/flow-per-strike-intraday": 2,
    "/api/stock/{ticker}/flow-per-strike": 3,
    "/api/stock/{ticker}/greek-exposure": 1,
    "/api/stock/{ticker}/greek-exposure/strike": 2,
    "/api/stock/{ticker}/greek-exposure/expiry": 3,
    "/api/stock/{ticker}/ohlc/{candle_size}": 1,
    "/api/darkpool/{ticker}": 1,
    "/api/lit-flow/{ticker}": 1,
    "/api/stock/{ticker}/volatility/term-structure": 1,
    "/api/stock/{ticker}/historical-risk-reversal-skew": 1
}

CALL_ALIASES = {"C", "CALL", "CALLS"}
PUT_ALIASES = {"P", "PUT", "PUTS"}
BULLISH_SIDE_ALIASES = {"ASK", "BUY", "BULL", "BULLISH", "BOT"}
BEARISH_SIDE_ALIASES = {"BID", "SELL", "BEAR", "BEARISH", "SOLD"}
EVENT_TIME_KEYS = (
    "event_time", "event_at", "executed_at", "trade_time", "occurred_at",
    "timestamp", "time", "t", "date", "updated_at", "last_updated"
)
PUBLISH_TIME_KEYS = (
    "source_publish_time", "published_at", "publish_time", "report_time", "report_date"
)
EFFECTIVE_TIME_KEYS = (
    "effective_at", "effective_ts", "effective_ts_utc", "effective_time", "as_of"
)
SOURCE_REVISION_KEYS = (
    "source_revision", "revision", "rev", "version", "sequence_id", "update_id"
)
SPOT_KEYS = ("spot", "underlying_price", "underlying", "stock_price", "spot_price")


FEATURE_USE_CONTRACT_VERSION = "feature_use/v1"
FEATURE_USE_ROLE_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "signal-critical": {
        "decision_path": True,
        "missing_affects_confidence": True,
        "stale_affects_confidence": True,
    },
    "context-only": {
        "decision_path": False,
        "missing_affects_confidence": False,
        "stale_affects_confidence": False,
    },
    "report-only": {
        "decision_path": False,
        "missing_affects_confidence": False,
        "stale_affects_confidence": False,
    },
    "disabled": {
        "decision_path": False,
        "missing_affects_confidence": False,
        "stale_affects_confidence": False,
    },
}
LEGACY_PATH_USE_ROLE_OVERRIDES: Dict[str, str] = {
    "/api/darkpool/{ticker}": "report-only",
}
OUTPUT_DOMAIN_CONTRACT_VERSION = "output_domain/v1"


def _normalize_expected_bounds(value: Any) -> Optional[Dict[str, Any]]:
    if value is None:
        return None

    lower: Optional[float] = None
    upper: Optional[float] = None
    inclusive = True
    if isinstance(value, Mapping):
        lower = safe_float(value.get("lower", value.get("min")))
        upper = safe_float(value.get("upper", value.get("max")))
        inclusive = bool(value.get("inclusive", True))
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        lower = safe_float(value[0])
        upper = safe_float(value[1])
    else:
        return None

    if lower is None or upper is None:
        return None
    if not math.isfinite(lower) or not math.isfinite(upper):
        return None

    lo = float(min(lower, upper))
    hi = float(max(lower, upper))
    return {"lower": lo, "upper": hi, "inclusive": inclusive}


def _with_output_domain(
    lineage: Dict[str, Any],
    *,
    emitted_units: Optional[str],
    raw_input_units: Optional[str],
    bounded_output: bool,
    expected_bounds: Any = None,
    output_domain: str = "unbounded_scalar",
    allowed_values: Optional[Iterable[Any]] = None,
) -> Dict[str, Any]:
    enriched = dict(lineage)
    enriched["emitted_units"] = emitted_units
    enriched["raw_input_units"] = raw_input_units
    enriched["bounded_output"] = bool(bounded_output)
    enriched["expected_bounds"] = _normalize_expected_bounds(expected_bounds) if expected_bounds is not None else None
    enriched["output_domain"] = output_domain
    enriched["output_domain_contract_version"] = OUTPUT_DOMAIN_CONTRACT_VERSION
    if allowed_values is not None:
        enriched["allowed_values"] = [
            float(v) if isinstance(v, (int, float)) and math.isfinite(float(v)) else v
            for v in allowed_values
        ]
    return enriched


def _resolve_feature_use_contract(ctx: EndpointContext) -> Dict[str, Any]:
    explicit_role = getattr(ctx, "endpoint_purpose", None)
    role_source = "endpoint_context"
    if explicit_role in FEATURE_USE_ROLE_DEFAULTS:
        use_role = str(explicit_role)
    else:
        fallback_role = LEGACY_PATH_USE_ROLE_OVERRIDES.get(ctx.path)
        if fallback_role in FEATURE_USE_ROLE_DEFAULTS:
            use_role = str(fallback_role)
            role_source = "path_fallback"
        else:
            use_role = "signal-critical"
            role_source = "default_signal_critical"

    defaults = FEATURE_USE_ROLE_DEFAULTS[use_role]
    raw_decision_path = getattr(ctx, "decision_path", None)
    raw_missing_affects_confidence = getattr(ctx, "missing_affects_confidence", None)
    raw_stale_affects_confidence = getattr(ctx, "stale_affects_confidence", None)

    decision_path = defaults["decision_path"] if raw_decision_path is None else bool(raw_decision_path)
    missing_affects_confidence = (
        defaults["missing_affects_confidence"] if raw_missing_affects_confidence is None else bool(raw_missing_affects_confidence)
    )
    stale_affects_confidence = (
        defaults["stale_affects_confidence"] if raw_stale_affects_confidence is None else bool(raw_stale_affects_confidence)
    )

    decision_eligible = bool(use_role == "signal-critical" and decision_path)
    if not decision_eligible:
        missing_affects_confidence = False
        stale_affects_confidence = False

    return {
        "contract_version": str(getattr(ctx, "purpose_contract_version", None) or FEATURE_USE_CONTRACT_VERSION),
        "use_role": use_role,
        "decision_path": decision_path,
        "decision_eligible": decision_eligible,
        "missing_affects_confidence": missing_affects_confidence,
        "stale_affects_confidence": stale_affects_confidence,
        "endpoint_name": getattr(ctx, "endpoint_name", None),
        "contract_source": role_source,
    }

def _normalize_signed(x: Optional[float], *, scale: float) -> Optional[float]:
    if scale == 0: 
        return None 
    val = safe_float(x)
    if val is None or not math.isfinite(val): 
        return None
    return max(-1.0, min(1.0, val / scale))

def _parse_strict_ts(row: dict, key: str) -> float:
    ts_val = row.get(key)
    if isinstance(ts_val, (int, float)): 
        return float(ts_val)
    if isinstance(ts_val, str):
        try:
            return datetime.datetime.fromisoformat(ts_val.replace('Z', '+00:00')).timestamp()
        except ValueError:
            pass
    return 0.0


def _serialize_dtlike(value: Any) -> Any:
    if isinstance(value, datetime.datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=datetime.timezone.utc)
        else:
            value = value.astimezone(datetime.timezone.utc)
        return value.isoformat()
    return value


def _coerce_ts_iso(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, datetime.datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=datetime.timezone.utc)
        else:
            value = value.astimezone(datetime.timezone.utc)
        return value.isoformat()
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            return None
        try:
            ts = float(value)
            abs_ts = abs(ts)

            # Some UW endpoints emit UNIX epoch timestamps in milliseconds (or higher precision).
            # Normalize to seconds so feature effective timestamps are not silently lost.
            if abs_ts > 1e14:
                ts = ts / (1e9 if abs_ts > 1e17 else 1e6)
            elif abs_ts > 1e11:
                ts = ts / 1e3

            dt = datetime.datetime.fromtimestamp(ts, datetime.timezone.utc)
            # Plausibility guard: avoid treating non-time numeric fields as timestamps.
            if dt.year < 1990 or dt.year > 2100:
                return None
            return dt.isoformat()
        except (OverflowError, OSError, ValueError):
            return None
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        try:
            dt = datetime.datetime.fromisoformat(raw.replace('Z', '+00:00'))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=datetime.timezone.utc)
            dt = dt.astimezone(datetime.timezone.utc)
            if dt.year < 1990 or dt.year > 2100:
                return None
            return dt.isoformat()
        except ValueError:
            try:
                return _coerce_ts_iso(float(raw))
            except (TypeError, ValueError, OverflowError, OSError):
                return None
    return None


def _iter_payload_dicts(payload: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(payload, dict):
        yield payload
    rows = grab_list(payload)
    if not rows and isinstance(payload, dict):
        rows = [payload]
    for row in rows:
        if isinstance(row, dict):
            yield row


def _find_payload_timestamp(payload: Any, keys: Tuple[str, ...]) -> Tuple[Optional[str], Optional[str], bool]:
    best_iso: Optional[str] = None
    best_dt: Optional[datetime.datetime] = None
    best_key: Optional[str] = None
    saw_candidate = False

    for item in _iter_payload_dicts(payload):
        for key in keys:
            if key not in item:
                continue
            saw_candidate = True
            iso_val = _coerce_ts_iso(item.get(key))
            if iso_val is None:
                continue
            dt_val = datetime.datetime.fromisoformat(iso_val)
            if best_dt is None or dt_val > best_dt:
                best_dt = dt_val
                best_iso = iso_val
                best_key = key

    if best_iso is not None:
        return best_iso, best_key, False
    if saw_candidate:
        return "INVALID", best_key, True
    return None, None, False


def _find_first_payload_value(payload: Any, keys: Tuple[str, ...]) -> Optional[Any]:
    for item in _iter_payload_dicts(payload):
        for key in keys:
            value = item.get(key)
            if value not in (None, ""):
                return value
    return None


def _extract_payload_provenance(payload: Any) -> Dict[str, Any]:
    event_iso, event_key, event_invalid = _find_payload_timestamp(payload, EVENT_TIME_KEYS)
    publish_iso, publish_key, publish_invalid = _find_payload_timestamp(payload, PUBLISH_TIME_KEYS)
    effective_iso, effective_key, effective_invalid = _find_payload_timestamp(payload, EFFECTIVE_TIME_KEYS)
    revision = _find_first_payload_value(payload, SOURCE_REVISION_KEYS)

    details: Dict[str, Any] = {}
    if effective_iso is not None:
        details["effective_at_utc"] = effective_iso
    elif event_iso is not None:
        details["effective_at_utc"] = event_iso
    elif publish_iso is not None:
        details["effective_at_utc"] = publish_iso
    elif effective_invalid or event_invalid or publish_invalid:
        details["effective_at_utc"] = "INVALID"

    if event_iso is not None:
        details["event_time_utc"] = event_iso
    elif event_invalid:
        details["event_time_utc"] = "INVALID"

    if publish_iso is not None:
        details["source_publish_time_utc"] = publish_iso
    elif publish_invalid:
        details["source_publish_time_utc"] = "INVALID"

    if revision is not None:
        details["source_revision"] = str(revision)

    if any(k is not None for k in (event_key, publish_key, effective_key)):
        details["payload_timestamp_keys"] = {
            "event_time": event_key,
            "source_publish_time": publish_key,
            "effective_at": effective_key,
        }
    return details


def _extract_spot_reference(payload: Any) -> Optional[float]:
    for item in _iter_payload_dicts(payload):
        for key in SPOT_KEYS:
            spot_val = safe_float(item.get(key))
            if spot_val is not None and math.isfinite(spot_val):
                return spot_val
    return None


def _normalize_balance(pos: float, neg: float) -> Optional[float]:
    total = pos + neg
    if total <= 0 or not math.isfinite(total):
        return None
    imbalance = (pos - neg) / total
    if not math.isfinite(imbalance):
        return None
    return max(-1.0, min(1.0, imbalance))


def _normalize_put_call(value: Any) -> Optional[str]:
    if value is None:
        return None
    norm = str(value).upper().strip()
    if norm in CALL_ALIASES:
        return "CALL"
    if norm in PUT_ALIASES:
        return "PUT"
    return None


def _extract_call_put_totals(payload: Any) -> Tuple[Optional[float], Optional[float], Dict[str, Any]]:
    """Best-effort extraction of aggregate call/put open-interest totals.

    Some vendor endpoints return OI as a single summary object (no per-contract rows and/or
    no put/call field). In those cases we can still compute a directional imbalance if we can
    locate aggregate call/put totals.
    """

    prov: Dict[str, Any] = {"totals_source": None, "call_total_key": None, "put_total_key": None}

    def _maybe_float(x: Any) -> Optional[float]:
        f = safe_float(x)
        if f is None or not math.isfinite(f):
            return None
        return f

    # Candidate containers to scan (shallow).
    containers: List[Tuple[str, Any]] = [("payload", payload)]
    if isinstance(payload, dict):
        for k in ("data", "result", "results", "summary", "totals"):
            if k in payload:
                containers.append((k, payload.get(k)))

    # Lowercase key candidates.
    call_keys = (
        "call_open_interest",
        "calls_open_interest",
        "call_oi",
        "calls_oi",
        "calloi",
        "callopeninterest",
        "call_openinterest",
        "callopen_interest",
        "calloi_total",
        "call_open_interest_total",
        "calloiTotal".lower(),
        "callOpenInterest".lower(),
    )
    put_keys = (
        "put_open_interest",
        "puts_open_interest",
        "put_oi",
        "puts_oi",
        "putoi",
        "putopeninterest",
        "put_openinterest",
        "putopen_interest",
        "putoi_total",
        "put_open_interest_total",
        "putoiTotal".lower(),
        "putOpenInterest".lower(),
    )

    def _scan_mapping(m: Dict[str, Any], label: str) -> Tuple[Optional[float], Optional[float], Optional[str], Optional[str]]:
        lower_map = {str(k).lower(): k for k in m.keys()}

        call_val = None
        call_key_used = None
        for ck in call_keys:
            k = lower_map.get(ck)
            if k is None:
                continue
            call_val = _maybe_float(m.get(k))
            if call_val is not None:
                call_key_used = str(k)
                break

        put_val = None
        put_key_used = None
        for pk in put_keys:
            k = lower_map.get(pk)
            if k is None:
                continue
            put_val = _maybe_float(m.get(k))
            if put_val is not None:
                put_key_used = str(k)
                break

        # Nested shape: {"calls": {"open_interest": ...}, "puts": {...}}
        if call_val is None:
            calls_obj = m.get(lower_map.get("calls")) or m.get(lower_map.get("call"))
            if isinstance(calls_obj, dict):
                nested = {str(k).lower(): k for k in calls_obj.keys()}
                for nk in ("open_interest", "openinterest", "oi"):
                    kk = nested.get(nk)
                    if kk is None:
                        continue
                    call_val = _maybe_float(calls_obj.get(kk))
                    if call_val is not None:
                        call_key_used = f"{lower_map.get('calls') or lower_map.get('call')}.{kk}"
                        break

        if put_val is None:
            puts_obj = m.get(lower_map.get("puts")) or m.get(lower_map.get("put"))
            if isinstance(puts_obj, dict):
                nested = {str(k).lower(): k for k in puts_obj.keys()}
                for nk in ("open_interest", "openinterest", "oi"):
                    kk = nested.get(nk)
                    if kk is None:
                        continue
                    put_val = _maybe_float(puts_obj.get(kk))
                    if put_val is not None:
                        put_key_used = f"{lower_map.get('puts') or lower_map.get('put')}.{kk}"
                        break

        return call_val, put_val, call_key_used, put_key_used

    for label, obj in containers:
        if not isinstance(obj, dict):
            continue
        call_val, put_val, call_key_used, put_key_used = _scan_mapping(obj, label)
        if call_val is not None and put_val is not None:
            prov.update({
                "totals_source": label,
                "call_total_key": call_key_used,
                "put_total_key": put_key_used,
            })
            return call_val, put_val, prov

    # Nothing found.
    prov["totals_source"] = "not_found"
    return None, None, prov

def _build_contract_normalization_failure_bundle(
    ctx: EndpointContext,
    extractor_name: str,
    lineage: Dict[str, Any],
    payload: Any,
    *,
    feature_keys: Iterable[str],
    meta_key: str,
    summary: Any,
) -> FeatureBundle:
    reason = getattr(summary, "failure_reason", None) or "contract_normalization_invalid"
    structured_log(
        logger,
        logging.WARNING,
        event="normalization_failure",
        msg="option contract normalization invalid",
        counter="normalization_failure_count",
        feature_key=meta_key,
        reason=reason,
        extractor=extractor_name,
    )
    meta = _build_meta(
        ctx,
        extractor_name,
        lineage,
        {
            **_extract_payload_provenance(payload),
            "status": "suppressed_contract_normalization_invalid",
            "suppression_reason": reason,
            "contract_normalization": summary.as_dict(),
        },
    )
    meta["na_reason"] = reason
    return FeatureBundle({k: None for k in feature_keys}, {meta_key: meta})


def _normalized_identity_for_row(contract_map: Mapping[int, Any], row_index: int) -> Optional[Any]:
    return contract_map.get(row_index)


def _build_meta(
    ctx: EndpointContext, 
    extractor_name: str, 
    lineage: Dict[str, Any], 
    details: Dict[str, Any] = None
) -> Dict[str, Any]:
    d = {"extractor": extractor_name}
    if details:
        d.update({_k: _serialize_dtlike(_v) for _k, _v in details.items()})

    explicit_effective = d.get("effective_at_utc", d.get("effective_ts_utc"))
    explicit_event_time = d.get("event_time_utc")
    explicit_publish_time = d.get("source_publish_time_utc")

    ctx_effective = _serialize_dtlike(getattr(ctx, "effective_ts_utc", None))
    ctx_event_time = _serialize_dtlike(getattr(ctx, "event_time_utc", None))
    ctx_publish_time = _serialize_dtlike(getattr(ctx, "source_publish_time_utc", None))
    received_at = d.get("received_at_utc") or _serialize_dtlike(getattr(ctx, "received_at_utc", None))
    processed_at = d.get("processed_at_utc") or _serialize_dtlike(getattr(ctx, "processed_at_utc", None))
    as_of_time = (
        d.get("as_of_time_utc")
        or _serialize_dtlike(getattr(ctx, "as_of_time_utc", None))
        or _serialize_dtlike(getattr(ctx, "endpoint_asof_ts_utc", None))
    )
    source_revision = d.get("source_revision") or getattr(ctx, "source_revision", None)

    eff_ts = None
    ts_source = getattr(ctx, "effective_time_source", None) or "missing_provider_time"
    ts_quality = getattr(ctx, "timestamp_quality", None) or "MISSING"

    if explicit_effective in ("INVALID", None) and explicit_effective is not None:
        eff_ts = None
        ts_source = d.get("effective_time_source") or "payload_effective_time"
        ts_quality = d.get("timestamp_quality") or "INVALID"
    elif explicit_effective:
        eff_ts = explicit_effective
        ts_source = d.get("effective_time_source") or "payload_effective_time"
        ts_quality = d.get("timestamp_quality") or "VALID"
    elif explicit_event_time in ("INVALID", None) and explicit_event_time is not None:
        eff_ts = None
        ts_source = "event_time"
        ts_quality = d.get("timestamp_quality") or "INVALID"
    elif explicit_event_time:
        eff_ts = explicit_event_time
        ts_source = "event_time"
        ts_quality = d.get("timestamp_quality") or "VALID"
    elif explicit_publish_time in ("INVALID", None) and explicit_publish_time is not None:
        eff_ts = None
        ts_source = "source_publish_time"
        ts_quality = d.get("timestamp_quality") or "INVALID"
    elif explicit_publish_time:
        eff_ts = explicit_publish_time
        ts_source = "source_publish_time"
        ts_quality = d.get("timestamp_quality") or "VALID"
    elif ctx_effective:
        eff_ts = ctx_effective
        ts_source = getattr(ctx, "effective_time_source", None) or "endpoint_provenance"
        ts_quality = getattr(ctx, "timestamp_quality", None) or "VALID"
    elif ctx_event_time:
        eff_ts = ctx_event_time
        ts_source = "event_time"
        ts_quality = getattr(ctx, "timestamp_quality", None) or "VALID"
    elif ctx_publish_time:
        eff_ts = ctx_publish_time
        ts_source = "source_publish_time"
        ts_quality = getattr(ctx, "timestamp_quality", None) or "VALID"

    event_time = explicit_event_time or ctx_event_time
    source_publish_time = explicit_publish_time or ctx_publish_time
    lagged = bool(d.get("lagged", getattr(ctx, "lagged", False)))
    time_provenance_degraded = bool(
        d.get("time_provenance_degraded", getattr(ctx, "time_provenance_degraded", False))
    )

    feature_use_contract = _resolve_feature_use_contract(ctx)
    structured_expected_bounds = _normalize_expected_bounds(lineage.get("expected_bounds"))
    allowed_values = lineage.get("allowed_values")
    if isinstance(allowed_values, tuple):
        allowed_values = list(allowed_values)
    full_lineage = {
        "metric_name": lineage.get("metric_name", "unknown"),
        "source_path": ctx.path,
        "fields_used": lineage.get("fields_used", []),
        "units_expected": lineage.get("units_expected", "unknown"),
        "emitted_units": lineage.get("emitted_units"),
        "raw_input_units": lineage.get("raw_input_units"),
        "expected_bounds": copy.deepcopy(structured_expected_bounds) if structured_expected_bounds is not None else None,
        "bounded_output": bool(lineage.get("bounded_output", False)),
        "output_domain": lineage.get("output_domain"),
        "output_domain_contract_version": lineage.get("output_domain_contract_version"),
        "allowed_values": copy.deepcopy(allowed_values) if allowed_values is not None else None,
        "normalization": lineage.get("normalization", "none"),
        "session_applicability": lineage.get("session_applicability", "PREMARKET/RTH/AFTERHOURS"),
        "quality_policy": lineage.get("quality_policy", "None on missing"),
        "criticality": lineage.get("criticality", "NON_CRITICAL"),
        "effective_ts_utc": eff_ts,
        "event_time": None if event_time == "INVALID" else event_time,
        "source_publish_time": None if source_publish_time == "INVALID" else source_publish_time,
        "received_at": received_at,
        "processed_at": processed_at,
        "effective_at": eff_ts,
        "as_of_time": as_of_time,
        "source_revision": source_revision,
        "timestamp_source": ts_source,
        "timestamp_quality": ts_quality,
        "lagged": lagged,
        "time_provenance_degraded": time_provenance_degraded,
        "decision_path_role": feature_use_contract["use_role"],
        "decision_eligible": feature_use_contract["decision_eligible"],
        "missing_affects_confidence": feature_use_contract["missing_affects_confidence"],
        "stale_affects_confidence": feature_use_contract["stale_affects_confidence"],
        "feature_use_contract_version": feature_use_contract["contract_version"],
    }

    return {
        "source_endpoints": [{
            "method": ctx.method,
            "path": ctx.path,
            "operation_id": ctx.operation_id,
            "endpoint_id": ctx.endpoint_id,
            "used_event_id": ctx.used_event_id,
            "signature": ctx.signature,
            "endpoint_name": getattr(ctx, "endpoint_name", None),
            "purpose": feature_use_contract["use_role"],
            "decision_path": feature_use_contract["decision_path"],
            "missing_affects_confidence": feature_use_contract["missing_affects_confidence"],
            "stale_affects_confidence": feature_use_contract["stale_affects_confidence"],
            "purpose_contract_version": feature_use_contract["contract_version"],
        }],
        "freshness_state": ctx.freshness_state,
        "stale_age_min": ctx.stale_age_min,
        "na_reason": ctx.na_reason,
        "feature_use_contract": feature_use_contract,
        "use_role": feature_use_contract["use_role"],
        "decision_eligible": feature_use_contract["decision_eligible"],
        "missing_affects_confidence": feature_use_contract["missing_affects_confidence"],
        "stale_affects_confidence": feature_use_contract["stale_affects_confidence"],
        "metric_lineage": full_lineage,
        "details": d
    }

def _build_error_meta(
    ctx: EndpointContext,
    extractor_name: str,
    lineage: Dict[str, Any],
    na_reason: str,
    details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    meta = _build_meta(ctx, extractor_name, lineage, details=details)
    meta["freshness_state"] = "ERROR"
    meta["na_reason"] = na_reason
    return meta


def _annotate_level_usage_contract(meta: Dict[str, Any], level_type: str) -> Dict[str, Any]:
    contract = copy.deepcopy(derived_level_usage_contract(level_type))
    meta["level_usage_contract"] = contract
    meta.setdefault("details", {}).setdefault("derived_level_contract", copy.deepcopy(contract))

    metric_lineage = meta.setdefault("metric_lineage", {})
    metric_lineage["decision_path_role"] = contract["decision_path_role"]
    metric_lineage["prediction_consumed"] = contract["prediction_consumed"]
    metric_lineage["contract_version"] = contract["contract_version"]
    metric_lineage["feature_contract_state"] = contract["feature_contract_state"]
    return meta

def extract_price_features(ohlc_payload: Any, ctx: EndpointContext) -> FeatureBundle:
    lineage = _with_output_domain({
        "metric_name": "spot",
        # UW OHLC candles are represented as Candle objects in their OpenAPI schema.
        # In live responses they commonly surface timestamps as `start_time`/`end_time`
        # (UTC ISO8601), but older/alternate shapes may use `t`/`timestamp`.
        "fields_used": ["close", "end_time", "start_time", "t", "timestamp", "time", "datetime", "date"],
        "units_expected": "USD",
        "normalization": "none",
        "session_applicability": "PREMARKET/RTH/AFTERHOURS",
        "quality_policy": "None if missing required explicit keys",
        "criticality": "CRITICAL"
    }, emitted_units="USD", raw_input_units="USD", bounded_output=False, output_domain="unbounded_scalar")
    
    if is_na(ohlc_payload) or ctx.freshness_state == "ERROR":
        return FeatureBundle({"spot": None}, {"price": _build_error_meta(ctx, "extract_price_features", lineage, ctx.na_reason or "missing_dependency")})
        
    rows = grab_list(ohlc_payload)
    if not rows:
        return FeatureBundle({"spot": None}, {"price": _build_error_meta(ctx, "extract_price_features", lineage, "no_rows")})

    def _row_ts(r: Dict[str, Any]) -> float:
        # Prefer close-of-candle timestamps when available.
        return (
            _parse_strict_ts(r, "end_time")
            or _parse_strict_ts(r, "start_time")
            or _parse_strict_ts(r, "t")
            or _parse_strict_ts(r, "timestamp")
            or _parse_strict_ts(r, "time")
            or _parse_strict_ts(r, "datetime")
            or _parse_strict_ts(r, "date")
            or 0.0
        )
    
    # As-of contract: when running an as-of snapshot (rounded bucket), the vendor payload can
    # include candles newer than ctx.as_of_time_utc (e.g., if the fetch happens a few minutes
    # after the bucket boundary). We must not select a candle whose timestamp is after the
    # snapshot as-of time.
    asof_ts: Optional[float] = None
    if getattr(ctx, "as_of_time_utc", None) is not None:
        try:
            asof_ts = float(ctx.as_of_time_utc.timestamp())
        except Exception:
            asof_ts = None

    eligible_rows = rows
    if asof_ts is not None:
        eligible_rows = [r for r in rows if (ts := _row_ts(r)) and ts <= asof_ts]
        if not eligible_rows:
            # If nothing is at-or-before as-of (e.g., as-of before market open), fall back to
            # the latest row to preserve prior behaviour while still surfacing the mismatch via
            # downstream as-of validation.
            eligible_rows = rows

    latest_row = max(eligible_rows, key=_row_ts)
    
    close_val = latest_row.get("close")
    # Prefer close-of-candle timestamps when available.
    t_val = (
        latest_row.get("end_time")
        if latest_row.get("end_time") is not None
        else latest_row.get("start_time")
        if latest_row.get("start_time") is not None
        else latest_row.get("t")
        if latest_row.get("t") is not None
        else latest_row.get("timestamp")
        if latest_row.get("timestamp") is not None
        else latest_row.get("time")
        if latest_row.get("time") is not None
        else latest_row.get("datetime")
        if latest_row.get("datetime") is not None
        else latest_row.get("date")
    )
    
    if close_val is None:
        return FeatureBundle({"spot": None}, {"price": _build_error_meta(ctx, "extract_price_features", lineage, "missing_explicit_close_field")})
        
    close_float = safe_float(close_val)
    ts_float = _row_ts(latest_row)

    # Normalize epoch-millis (and other high precision epochs) and guard implausible values.
    eff_ts = _coerce_ts_iso(ts_float)
    if eff_ts is None and t_val is not None:
        # Timestamp field exists but could not be parsed/coerced.
        eff_ts = "INVALID"
    prov = _extract_payload_provenance(latest_row)
    prov.update({
        "last_ts": t_val,
        "effective_at_utc": eff_ts,
        "event_time_utc": None if eff_ts == "INVALID" else eff_ts,
    })

    return FeatureBundle({"spot": close_float}, {"price": _build_meta(ctx, "extract_price_features", lineage, prov)})

def extract_smart_whale_pressure(flow_payload: Any, ctx: EndpointContext, min_premium: float = 10000.0, max_dte: float = 14.0, norm_scale: float = 500_000.0) -> FeatureBundle:
    lineage = _with_output_domain({
        "metric_name": "smart_whale_pressure",
        "fields_used": ["premium", "dte", "side", "put_call", "option_symbol", "expiration", "multiplier", "deliverable"],
        "units_expected": "Normalized Directional Pressure [-1, 1]",
        "normalization": f"normalize_signed [-1, 1] by {norm_scale}; require canonical contract normalization for contract-level rows",
        "session_applicability": "RTH",
        "quality_policy": "None on filtered zeros to avoid false baseline certainty; suppress on contract normalization failure",
        "criticality": "CRITICAL"
    }, emitted_units="normalized_directional_pressure", raw_input_units="Net Premium Flow (USD)", bounded_output=True, expected_bounds=(-1.0, 1.0), output_domain="closed_interval")

    if is_na(flow_payload) or ctx.freshness_state == "ERROR":
        return FeatureBundle({"smart_whale_pressure": None}, {"flow": _build_error_meta(ctx, "extract_smart_whale_pressure", lineage, ctx.na_reason or "missing_dependency")})

    trades = grab_list(flow_payload)
    if not trades and isinstance(flow_payload, dict) and "data" in flow_payload:
        trades = flow_payload["data"]

    if trades and not all(isinstance(t, dict) for t in trades):
        return FeatureBundle({"smart_whale_pressure": None}, {"flow": _build_error_meta(ctx, "extract_smart_whale_pressure", lineage, "schema_non_dict_rows")})

    if not trades:
        # Neutral (0.0) when the endpoint returns an empty set.
        #
        # Rationale: returning None here can hard-block LIVE predictions when flow endpoints
        # intermittently return empty payloads (common near open or for quieter tickers).
        # A neutral value preserves safety (no directional signal) while meta marks the
        # confidence impact as degraded.
        meta = _build_meta(
            ctx,
            "extract_smart_whale_pressure",
            lineage,
            {
                **_extract_payload_provenance(flow_payload),
                "status": "computed_neutral_from_empty",
                "n_trades": 0,
                "confidence_impact": "DEGRADED",
            },
        )
        meta["freshness_state"] = "EMPTY_VALID"
        meta["na_reason"] = "empty_payload"
        return FeatureBundle({"smart_whale_pressure": 0.0}, {"flow": meta})

    norm_summary = normalize_option_rows(trades)
    if norm_summary.status == "INVALID":
        return _build_contract_normalization_failure_bundle(
            ctx,
            "extract_smart_whale_pressure",
            lineage,
            flow_payload,
            feature_keys=["smart_whale_pressure"],
            meta_key="flow",
            summary=norm_summary,
        )
    contract_map = normalized_contract_map(norm_summary)

    whale_call, whale_put = 0.0, 0.0
    valid_count, skip_missing_fields, skip_bad_type, skip_bad_side, skip_threshold = 0, 0, 0, 0, 0

    pc_map = {"C": "CALL", "CALL": "CALL", "CALLS": "CALL", "P": "PUT", "PUT": "PUT", "PUTS": "PUT"}
    side_map = {"ASK": "BULL", "BUY": "BULL", "BULLISH": "BULL", "BOT": "BULL", "BID": "BEAR", "SELL": "BEAR", "BEARISH": "BEAR", "SOLD": "BEAR"}

    for idx, t in enumerate(trades):
        prem = safe_float(t.get("premium"))
        dte = safe_float(t.get("dte"))
        side_raw = t.get("side")
        identity = _normalized_identity_for_row(contract_map, idx)
        pc_raw = identity.put_call if identity is not None else (t.get("put_call") or t.get("option_type") or t.get("type") or t.get("pc"))

        if prem is None or dte is None or is_na(side_raw) or is_na(pc_raw):
            skip_missing_fields += 1
            continue

        pc_norm = pc_raw if identity is not None else pc_map.get(str(pc_raw).upper().strip())
        side_norm = side_map.get(str(side_raw).upper().strip())

        if not pc_norm:
            skip_bad_type += 1
            continue
        if not side_norm:
            skip_bad_side += 1
            continue

        if prem < min_premium or dte > max_dte:
            skip_threshold += 1
            continue

        valid_count += 1
        if side_norm == "BULL":
            if pc_norm == "CALL":
                whale_call += prem
            else:
                whale_put += prem
        elif side_norm == "BEAR":
            if pc_norm == "CALL":
                whale_call -= prem
            else:
                whale_put -= prem

    details = {
        **_extract_payload_provenance(flow_payload),
        "contract_normalization": norm_summary.as_dict(),
        "n_raw_trades": len(trades),
    }

    if valid_count == 0:
        details.update({
            "status": "computed_neutral_no_trades_met_thresholds",
            "skipped_threshold": skip_threshold,
            "confidence_impact": "DEGRADED",
        })
        meta = _build_meta(ctx, "extract_smart_whale_pressure", lineage, details)
        meta["freshness_state"] = "EMPTY_VALID"
        meta["na_reason"] = "no_trades_met_policy_thresholds"
        return FeatureBundle({"smart_whale_pressure": 0.0}, {"flow": meta})

    net = whale_call - whale_put
    details.update({"net_prem": net, "n_valid": valid_count})
    meta = _build_meta(ctx, "extract_smart_whale_pressure", lineage, details)
    return FeatureBundle({"smart_whale_pressure": _normalize_signed(net, scale=norm_scale)}, {"flow": meta})

def extract_dealer_greeks(greek_payload: Any, ctx: EndpointContext, norm_scale: float = 1_000_000_000.0) -> FeatureBundle:
    keys = ["dealer_vanna", "dealer_charm", "net_gamma_exposure_notional"]
    lineage = _with_output_domain({
        "metric_name": "dealer_greeks",
        "fields_used": ["vanna_exposure", "charm_exposure", "gamma_exposure", "date", "option_symbol", "expiration", "multiplier", "deliverable"],
        "units_expected": "Normalized Signed Exposure [-1, 1]",
        "normalization": f"normalize_signed [-1, 1] by {norm_scale}; require canonical contract normalization for contract-level rows",
        "session_applicability": "PREMARKET/RTH",
        "quality_policy": "None on missing; suppress on contract normalization failure",
        "criticality": "CRITICAL"
    }, emitted_units="normalized_signed_exposure", raw_input_units="Notional Exposure (USD)", bounded_output=True, expected_bounds=(-1.0, 1.0), output_domain="closed_interval")

    if is_na(greek_payload) or ctx.freshness_state == "ERROR":
        return FeatureBundle({k: None for k in keys}, {"greeks": _build_error_meta(ctx, "extract_dealer_greeks", lineage, ctx.na_reason or "missing_dependency")})

    rows = grab_list(greek_payload)
    if not rows:
        return FeatureBundle({k: None for k in keys}, {"greeks": _build_error_meta(ctx, "extract_dealer_greeks", lineage, "no_rows")})

    norm_summary = normalize_option_rows(rows)
    if norm_summary.status == "INVALID":
        return _build_contract_normalization_failure_bundle(
            ctx,
            "extract_dealer_greeks",
            lineage,
            greek_payload,
            feature_keys=keys,
            meta_key="greeks",
            summary=norm_summary,
        )
    contract_map = normalized_contract_map(norm_summary)

    latest_idx, latest = max(enumerate(rows), key=lambda item: _parse_strict_ts(item[1], "date"))

    # The upstream greeks payload frequently provides a *date* marker (YYYY-MM-DD) rather than a
    # precise provider event/publish timestamp. Treating that date as an event/effective timestamp
    # produces midnight UTC effective_ts values that violate join-skew/freshness rules after market
    # open/close.
    #
    # We therefore:
    #   1) Use the row date only for row selection + audit metadata.
    #   2) Delegate the feature effective timestamp to EndpointTruth (ctx.effective_ts_utc).
    ts_float = _parse_strict_ts(latest, "date")
    date_raw = latest.get("date")
    row_ts_utc = datetime.datetime.fromtimestamp(ts_float, datetime.timezone.utc).isoformat() if ts_float > 0 else None

    prov: Dict[str, Any] = {}
    prov.update({
        "candidate_row_count": len(rows),
        "selected_row_index": latest_idx,
        "payload_row_date_raw": date_raw,
        "payload_row_date_utc": row_ts_utc,
        "ignored_payload_date_for_effective_ts": True,
        "scale_used": norm_scale,
        "contract_normalization": norm_summary.as_dict(),
    })

    # If EndpointTruth could not infer a true snapshot timestamp, allow a deterministic fallback
    # for date-only payloads: midnight UTC of the provided YYYY-MM-DD. This prevents downstream
    # freshness gating from treating the feature as missing time provenance entirely.
    if ctx.effective_ts_utc is None:
        inferred_snapshot_ts_utc = _infer_daily_snapshot_effective_ts_utc(greek_payload)
        if inferred_snapshot_ts_utc is not None:
            prov.update({
                "effective_ts_utc": inferred_snapshot_ts_utc,
                "effective_time_source": "payload_date_midnight_utc",
                "timestamp_quality": "DATE_ONLY",
                "time_provenance_degraded": True,
            })
    identity = _normalized_identity_for_row(contract_map, latest_idx)
    if identity is not None:
        prov["canonical_contract_key"] = identity.canonical_contract_key
    meta = _build_meta(ctx, "extract_dealer_greeks", lineage, prov)

    return FeatureBundle({
        "dealer_vanna": _normalize_signed(safe_float(latest.get("vanna_exposure")), scale=norm_scale),
        "dealer_charm": _normalize_signed(safe_float(latest.get("charm_exposure")), scale=norm_scale),
        "net_gamma_exposure_notional": _normalize_signed(safe_float(latest.get("gamma_exposure")), scale=norm_scale)
    }, {"greeks": meta})

def extract_gex_sign(spot_exposures_payload: Any, ctx: EndpointContext) -> FeatureBundle:
    lineage = _with_output_domain({
        "metric_name": "net_gex_sign",
        # UW payloads for spot exposures are not fully stable; different endpoints / versions
        # have been observed to emit the aggregate gamma exposure under different keys.
        "fields_used": ["gamma_exposure", "gex", "gamma", "total_gamma", "net_gamma", "net_gex"],
        "units_expected": "Directional Sign [-1, 1]",
        "normalization": "Directional sign clamping",
        "session_applicability": "PREMARKET/RTH",
        "quality_policy": "None on missing exposure fields",
        "criticality": "CRITICAL"
    }, emitted_units="directional_sign", raw_input_units="Gamma Exposure (provider aggregate units)", bounded_output=True, expected_bounds=(-1.0, 1.0), output_domain="discrete_sign", allowed_values=(-1.0, 0.0, 1.0))
    
    if is_na(spot_exposures_payload) or ctx.freshness_state == "ERROR":
        return FeatureBundle({"net_gex_sign": None}, {"gex": _build_error_meta(ctx, "extract_gex_sign", lineage, ctx.na_reason or "missing_dependency")})

    rows = grab_list(spot_exposures_payload)
    if not rows and isinstance(spot_exposures_payload, dict):
        # Some vendor responses return a single dict row (sometimes nested under wrapper keys).
        for k in ("data", "result", "results", "item", "summary"):
            v = spot_exposures_payload.get(k)
            if isinstance(v, dict):
                rows = [v]
                break
        if not rows:
            rows = [spot_exposures_payload]
    if not rows:
        return FeatureBundle({"net_gex_sign": None}, {"gex": _build_error_meta(ctx, "extract_gex_sign", lineage, "no_rows")})

    # Keys we attempt (in priority order) for gamma exposure.
    # Vendor responses vary across endpoints; we support both snake_case and camelCase.
    gex_aliases = (
        # snake_case
        "net_gamma_exposure",
        "net_gamma_exposure_notional",
        "net_gamma",
        "net_gex",
        "net_gex_notional",
        "gamma_exposure",
        "total_gamma",
        "gamma",
        "gex",
        # camelCase variants
        "netGammaExposure",
        "netGammaExposureNotional",
        "netGamma",
        "netGex",
        "netGexNotional",
        "gammaExposure",
    )

    def _extract_gex_value(row: Dict[str, Any]) -> Optional[float]:
        lower_map = {str(k).lower(): k for k in row.keys()}
        for alias in gex_aliases:
            k = lower_map.get(alias.lower())
            if k is None:
                continue
            v = row.get(k)
            # Accept 0 values as valid.
            if v is None:
                continue
            if isinstance(v, str) and v.strip() == "":
                continue
            f = safe_float(v)
            if f is not None:
                return f

        # Heuristic fallback: a numeric field that looks like a net gamma exposure.
        for k, v in row.items():
            lk = str(k).lower()
            if "net" in lk and ("gex" in lk or "gamma" in lk) and ("exposure" in lk or "notional" in lk):
                f = safe_float(v)
                if f is not None:
                    return f
        return None
    
    tot_gamma = 0.0
    valid_rows = 0
    for r in rows:
        g = _extract_gex_value(r)
        if g is not None:
            tot_gamma += g
            valid_rows += 1
            
    if valid_rows == 0:
        # Keep the error reason stable for existing dashboards, but attach helpful diagnostics.
        sample_keys = sorted({k for r in rows[:3] for k in (r or {}).keys()})
        return FeatureBundle({"net_gex_sign": None}, {"gex": _build_error_meta(
            ctx,
            "extract_gex_sign",
            lineage,
            "missing_gamma_exposure_fields",
            details={
                "attempted_field_aliases": list(gex_aliases),
                "sample_row_keys": sample_keys,
                "sample_row_count": min(3, len(rows)),
                "total_row_count": len(rows),
            }
        )})
    
    if abs(tot_gamma) <= 1e-9: 
        sign = 0.0
    else: 
        sign = 1.0 if tot_gamma > 0 else -1.0

    meta = _build_meta(ctx, "extract_gex_sign", lineage, {**_extract_payload_provenance(spot_exposures_payload), "total": tot_gamma, "n_strikes": valid_rows})
    return FeatureBundle({"net_gex_sign": sign}, {"gex": meta})

def extract_oi_features(payload: Any, ctx: EndpointContext) -> FeatureBundle:
    lineage = _with_output_domain({
        "metric_name": "oi_pressure",
        "fields_used": [
            "open_interest",
            "oi",
            "strike",
            "strike_price",
            # put/call semantics (provider varies)
            "put_call",
            "option_type",
            "type",
            "pc",
            "right",
            "call_put",
            "callPut",
            "side",
            "cp",
            "putCall",
            "optionType",
            "is_call",
            "is_put",
            "direction",
            # contract-normalization aids
            "expiration",
            "option_symbol",
            "multiplier",
            "deliverable",
        ],
        "units_expected": "Directional Imbalance Ratio [-1, 1]",
        "normalization": "(weighted_call_equivalent_oi - weighted_put_equivalent_oi) / (weighted_call_equivalent_oi + weighted_put_equivalent_oi) with canonical contract normalization for contract-level rows",
        "session_applicability": "RTH",
        "quality_policy": "Suppress on missing put/call semantics, invalid directional rows, or contract normalization failure",
        "criticality": "CRITICAL"
    }, emitted_units="directional_imbalance_ratio", raw_input_units="Open Interest (contracts)", bounded_output=True, expected_bounds=(-1.0, 1.0), output_domain="closed_interval")
    if is_na(payload) or ctx.freshness_state == "ERROR":
        return FeatureBundle({"oi_pressure": None}, {"oi": _build_error_meta(ctx, "extract_oi", lineage, ctx.na_reason or "missing_dependency")})

    rows = grab_list(payload)
    if not rows:
        # Fallback: some payloads only provide aggregate call/put totals.
        call_total, put_total, totals_prov = _extract_call_put_totals(payload)
        if call_total is not None and put_total is not None:
            pressure = _normalize_balance(call_total, put_total)
            meta = _build_meta(
                ctx,
                "extract_oi",
                lineage,
                {
                    **_extract_payload_provenance(payload),
                    **totals_prov,
                    "status": "computed_from_summary_totals",
                    "call_total": call_total,
                    "put_total": put_total,
                    "weighting": "summary_call_put_totals",
                },
            )
            return FeatureBundle({"oi_pressure": pressure}, {"oi": meta})

        return FeatureBundle({"oi_pressure": None}, {"oi": _build_error_meta(ctx, "extract_oi", lineage, "no_rows")})

    norm_summary = normalize_option_rows(rows)
    if norm_summary.status == "INVALID":
        return _build_contract_normalization_failure_bundle(
            ctx,
            "extract_oi",
            lineage,
            payload,
            feature_keys=["oi_pressure"],
            meta_key="oi",
            summary=norm_summary,
        )
    contract_map = normalized_contract_map(norm_summary)

    spot_ref = _extract_spot_reference(payload)

    # Provider schemas vary; attempt multiple key aliases for call/put semantics.
    pc_key_aliases = (
        "put_call",
        "option_type",
        "type",
        "pc",
        "right",
        "call_put",
        "callPut",
        "side",
        "cp",
        "putCall",
        "optionType",
        "direction",
    )

    def _extract_put_call_raw(row: Dict[str, Any]) -> Any:
        # Boolean flags (seen in some vendor payloads)
        if "is_call" in row and row.get("is_call") is not None:
            return "CALL" if bool(row.get("is_call")) else "PUT"
        if "is_put" in row and row.get("is_put") is not None:
            return "PUT" if bool(row.get("is_put")) else "CALL"

        for k in pc_key_aliases:
            if k not in row:
                continue
            v = row.get(k)
            if v is None:
                continue
            if isinstance(v, str) and v.strip() == "":
                continue
            return v
        return None
    weighted_call_oi = 0.0
    weighted_put_oi = 0.0
    directional_rows = 0
    parsed_rows = 0
    missing_put_call_rows = 0
    normalized_contract_rows = 0

    for idx, r in enumerate(rows):
        identity = _normalized_identity_for_row(contract_map, idx)
        oi_val = safe_float(r.get("open_interest") or r.get("oi"))
        strike = identity.strike if identity is not None else safe_float(r.get("strike") or r.get("strike_price"))

        # Prefer contract-derived put/call when present; otherwise fall back to row aliases.
        pc_norm = None
        if identity is not None and identity.put_call:
            pc_norm = identity.put_call
        if pc_norm is None:
            pc_norm = _normalize_put_call(_extract_put_call_raw(r))

        if oi_val is None or not math.isfinite(oi_val) or oi_val < 0:
            continue
        parsed_rows += 1
        if pc_norm is None:
            missing_put_call_rows += 1
            continue

        weight = 1.0
        if spot_ref is not None and strike is not None and math.isfinite(strike):
            spot_band = max(abs(spot_ref) * 0.02, 1.0)
            weight = 1.0 / (1.0 + abs(strike - spot_ref) / spot_band)

        scale = contract_scale(identity) if identity is not None else 1.0
        equivalent_oi = oi_val * scale
        if identity is not None:
            normalized_contract_rows += 1

        directional_rows += 1
        if pc_norm == "CALL":
            weighted_call_oi += equivalent_oi * weight
        else:
            weighted_put_oi += equivalent_oi * weight

    if directional_rows == 0:
        # Fallback: attempt to compute from summary totals if available.
        call_total, put_total, totals_prov = _extract_call_put_totals(payload)
        if call_total is not None and put_total is not None:
            pressure = _normalize_balance(call_total, put_total)
            meta = _build_meta(ctx, "extract_oi", lineage, {
                **_extract_payload_provenance(payload),
                "contract_normalization": norm_summary.as_dict(),
                "n_rows": len(rows),
                "parsed_rows": parsed_rows,
                "missing_put_call_rows": missing_put_call_rows,
                "normalized_contract_rows": normalized_contract_rows,
                **totals_prov,
                "status": "computed_from_summary_totals_fallback",
                "call_total": call_total,
                "put_total": put_total,
                "spot_reference": spot_ref,
                "weighting": "summary_call_put_totals",
            })
            return FeatureBundle({"oi_pressure": pressure}, {"oi": meta})

        logger.warning(
            "unsafe_oi_pressure_suppressed",
            extra={"counter": "unsafe_directional_metric_suppressed", "feature_key": "oi_pressure", "reason": "missing_put_call_or_directional_rows"},
        )
        meta = _build_meta(ctx, "extract_oi", lineage, {
            **_extract_payload_provenance(payload),
            "contract_normalization": norm_summary.as_dict(),
            "n_rows": len(rows),
            "parsed_rows": parsed_rows,
            "missing_put_call_rows": missing_put_call_rows,
            "normalized_contract_rows": normalized_contract_rows,
            "status": "suppressed_directionless_oi_total",
            "suppression_reason": "missing_put_call_or_directional_rows",
            "spot_reference": spot_ref,
            "attempted_put_call_aliases": list(pc_key_aliases),
            "sample_row_keys": sorted({k for rr in rows[:3] for k in (rr or {}).keys()}),
        })
        meta["na_reason"] = "missing_put_call_or_directional_rows"
        return FeatureBundle({"oi_pressure": None}, {"oi": meta})

    pressure = _normalize_balance(weighted_call_oi, weighted_put_oi)
    meta = _build_meta(ctx, "extract_oi", lineage, {
        **_extract_payload_provenance(payload),
        "contract_normalization": norm_summary.as_dict(),
        "n_rows": len(rows),
        "parsed_rows": parsed_rows,
        "directional_rows": directional_rows,
        "normalized_contract_rows": normalized_contract_rows,
        "weighted_call_oi": weighted_call_oi,
        "weighted_put_oi": weighted_put_oi,
        "spot_reference": spot_ref,
        "weighting": "near_spot_inverse_distance" if spot_ref is not None else "unweighted_call_put_balance",
    })
    return FeatureBundle({"oi_pressure": pressure}, {"oi": meta})

def extract_volatility_features(payload: Any, ctx: EndpointContext) -> FeatureBundle:
    lineage = _with_output_domain({
        "metric_name": "iv_rank",
        "fields_used": ["iv_rank", "iv_percentile"],
        "units_expected": "Percentile [0, 1]",
        "normalization": "none",
        "session_applicability": "PREMARKET/RTH/AFTERHOURS",
        "quality_policy": "None on missing",
        "criticality": "NON_CRITICAL"
    }, emitted_units="percentile_rank", raw_input_units="Percentile Rank", bounded_output=True, expected_bounds=(0.0, 1.0), output_domain="closed_interval")
    if is_na(payload) or ctx.freshness_state == "ERROR":
        return FeatureBundle({"iv_rank": None}, {"vol": _build_error_meta(ctx, "extract_vol", lineage, ctx.na_reason or "missing_dependency")})
    
    rows = grab_list(payload)
    if not rows and isinstance(payload, dict):
        rows = [payload]
        
    val = safe_float(rows[0].get("iv_rank")) if rows else None
    if val is None or not math.isfinite(val):
        return FeatureBundle({"iv_rank": None}, {"vol": _build_error_meta(ctx, "extract_vol", lineage, "missing_iv_rank")})
    
    return FeatureBundle({"iv_rank": val}, {"vol": _build_meta(ctx, "extract_vol", lineage, _extract_payload_provenance(payload))})

def extract_vol_term_structure(payload: Any, ctx: EndpointContext) -> FeatureBundle:
    lineage = _with_output_domain({
        "metric_name": "vol_term_slope",
        "fields_used": ["dte", "days", "iv", "implied_volatility"],
        "units_expected": "IV Spread",
        "normalization": "none",
        "session_applicability": "PREMARKET/RTH/AFTERHOURS",
        "quality_policy": "None on missing",
        "criticality": "NON_CRITICAL"
    }, emitted_units="implied_volatility_spread", raw_input_units="Implied Volatility", bounded_output=False, output_domain="unbounded_scalar")
    if is_na(payload) or ctx.freshness_state == "ERROR":
        return FeatureBundle({"vol_term_slope": None}, {"vol_ts": _build_error_meta(ctx, "extract_vol_term_structure", lineage, ctx.na_reason or "missing_dependency")})
    
    rows = grab_list(payload)
    if not rows:
        return FeatureBundle({"vol_term_slope": None}, {"vol_ts": _build_error_meta(ctx, "extract_vol_term_structure", lineage, "no_rows")})
        
    valid_pts = []
    for r in rows:
        d = safe_float(r.get("dte") or r.get("days"))
        iv = safe_float(r.get("iv") or r.get("implied_volatility") or r.get("value"))
        if d is not None and iv is not None and math.isfinite(d) and math.isfinite(iv):
            valid_pts.append((d, iv))
    
    if len(valid_pts) < 2:
        return FeatureBundle({"vol_term_slope": None}, {"vol_ts": _build_error_meta(ctx, "extract_vol_term_structure", lineage, "insufficient_data_points")})
        
    valid_pts.sort(key=lambda x: x[0])
    slope = valid_pts[-1][1] - valid_pts[0][1]

    prov = {**_extract_payload_provenance(payload), "n_rows": len(valid_pts)}
    if ctx.effective_ts_utc is None:
        inferred_snapshot_ts_utc = _infer_daily_snapshot_effective_ts_utc(payload)
        if inferred_snapshot_ts_utc is not None:
            prov.update({
                "effective_ts_utc": inferred_snapshot_ts_utc,
                "effective_time_source": "payload_date_midnight_utc",
                "timestamp_quality": "DATE_ONLY",
                "time_provenance_degraded": True,
            })

    return FeatureBundle({"vol_term_slope": slope}, {"vol_ts": _build_meta(ctx, "extract_vol_term_structure", lineage, prov)})

def extract_vol_skew(payload: Any, ctx: EndpointContext) -> FeatureBundle:
    lineage = _with_output_domain({
        "metric_name": "vol_skew",
        "fields_used": ["skew", "risk_reversal", "value"],
        "units_expected": "Skew Ratio",
        "normalization": "none",
        "session_applicability": "PREMARKET/RTH/AFTERHOURS",
        "quality_policy": "None on missing",
        "criticality": "NON_CRITICAL"
    }, emitted_units="skew_ratio", raw_input_units="Skew Ratio", bounded_output=False, output_domain="unbounded_scalar")
    if is_na(payload) or ctx.freshness_state == "ERROR":
        return FeatureBundle({"vol_skew": None}, {"skew": _build_error_meta(ctx, "extract_vol_skew", lineage, ctx.na_reason or "missing_dependency")})
    
    rows = grab_list(payload)
    if not rows and isinstance(payload, dict):
        rows = [payload]
        
    if not rows:
        return FeatureBundle({"vol_skew": None}, {"skew": _build_error_meta(ctx, "extract_vol_skew", lineage, "no_rows")})
        
    latest = rows[0]
    skew_val = safe_float(latest.get("skew") or latest.get("risk_reversal") or latest.get("value"))
    
    if skew_val is None or not math.isfinite(skew_val):
        return FeatureBundle({"vol_skew": None}, {"skew": _build_error_meta(ctx, "extract_vol_skew", lineage, "missing_or_invalid_skew")})
        
    return FeatureBundle({"vol_skew": skew_val}, {"skew": _build_meta(ctx, "extract_vol_skew", lineage, _extract_payload_provenance(payload))})

def extract_darkpool_pressure(payload: Any, ctx: EndpointContext) -> FeatureBundle:
    lineage = _with_output_domain({
        "metric_name": "darkpool_pressure",
        "fields_used": ["volume", "price", "size", "side"],
        "units_expected": "Directional Imbalance Ratio [-1, 1]",
        "normalization": "Suppress unless explicit side semantics exist; otherwise (buy_notional - sell_notional) / total_notional",
        "session_applicability": "PREMARKET/RTH/AFTERHOURS",
        "quality_policy": "Suppress on directionless totals",
        "criticality": "NON_CRITICAL"
    }, emitted_units="directional_imbalance_ratio", raw_input_units="Notional Flow (USD)", bounded_output=True, expected_bounds=(-1.0, 1.0), output_domain="closed_interval")
    if is_na(payload) or ctx.freshness_state == "ERROR":
        return FeatureBundle({"darkpool_pressure": None}, {"darkpool": _build_error_meta(ctx, "extract_darkpool", lineage, ctx.na_reason or "missing_dependency")})
    
    rows = grab_list(payload)
    if not rows:
        return FeatureBundle({"darkpool_pressure": None}, {"darkpool": _build_error_meta(ctx, "extract_darkpool", lineage, "no_rows")})

    buy_notional = 0.0
    sell_notional = 0.0
    explicit_side_rows = 0
    valid_rows = 0
    for r in rows:
        vol = safe_float(r.get("volume", 0)) or safe_float(r.get("size", 0)) or 0.0
        price = safe_float(r.get("price", 0)) or 0.0
        notional = vol * price
        if not math.isfinite(notional):
            continue
        valid_rows += 1
        side = str(r.get("side", "")).upper().strip()
        if side in BULLISH_SIDE_ALIASES:
            buy_notional += notional
            explicit_side_rows += 1
        elif side in BEARISH_SIDE_ALIASES:
            sell_notional += notional
            explicit_side_rows += 1

    if explicit_side_rows == 0:
        logger.warning(
            "unsafe_darkpool_pressure_suppressed",
            extra={"counter": "unsafe_directional_metric_suppressed", "feature_key": "darkpool_pressure", "reason": "directionless_total"},
        )
        meta = _build_meta(ctx, "extract_darkpool", lineage, {
            **_extract_payload_provenance(payload),
            "n_rows": len(rows),
            "valid_rows": valid_rows,
            "status": "suppressed_directionless_darkpool_total",
            "suppression_reason": "directionless_total",
        })
        meta["na_reason"] = "directionless_total"
        return FeatureBundle({"darkpool_pressure": None}, {"darkpool": meta})

    pressure = _normalize_balance(buy_notional, sell_notional)
    return FeatureBundle({"darkpool_pressure": pressure}, {"darkpool": _build_meta(ctx, "extract_darkpool", lineage, {**_extract_payload_provenance(payload), "n_rows": len(rows), "explicit_side_rows": explicit_side_rows, "buy_notional": buy_notional, "sell_notional": sell_notional})})

def extract_litflow_pressure(payload: Any, ctx: EndpointContext) -> FeatureBundle:
    lineage = _with_output_domain({
        "metric_name": "litflow_pressure",
        "fields_used": ["volume", "price", "side", "size"],
        "units_expected": "Directional Imbalance Ratio [-1, 1]",
        "normalization": "(buy_notional - sell_notional) / total_notional",
        "session_applicability": "RTH",
        "quality_policy": "None on missing",
        "criticality": "NON_CRITICAL"
    }, emitted_units="directional_imbalance_ratio", raw_input_units="Notional Flow (USD)", bounded_output=True, expected_bounds=(-1.0, 1.0), output_domain="closed_interval")
    if is_na(payload) or ctx.freshness_state == "ERROR":
        return FeatureBundle({"litflow_pressure": None}, {"litflow": _build_error_meta(ctx, "extract_litflow", lineage, ctx.na_reason or "missing_dependency")})
    
    rows = grab_list(payload)
    if not rows:
        return FeatureBundle({"litflow_pressure": None}, {"litflow": _build_error_meta(ctx, "extract_litflow", lineage, "no_rows")})

    buy_notional = 0.0
    sell_notional = 0.0
    side_rows = 0
    for r in rows:
        vol = safe_float(r.get("volume", 0)) or safe_float(r.get("size", 0)) or 0.0
        price = safe_float(r.get("price", 0)) or 0.0
        side = str(r.get("side", "")).upper()
        
        notional = vol * price
        if not math.isfinite(notional):
            continue
            
        if side in BULLISH_SIDE_ALIASES:
            buy_notional += notional
            side_rows += 1
        elif side in BEARISH_SIDE_ALIASES:
            sell_notional += notional
            side_rows += 1

    pressure = _normalize_balance(buy_notional, sell_notional)
    meta = _build_meta(ctx, "extract_litflow", lineage, {
        **_extract_payload_provenance(payload),
        "n_rows": len(rows),
        "side_rows": side_rows,
        "buy_notional": buy_notional,
        "sell_notional": sell_notional,
    })
    return FeatureBundle({"litflow_pressure": pressure}, {"litflow": meta})

EXTRACTOR_REGISTRY = {
    "/api/stock/{ticker}/spot-exposures": "GEX",
    "/api/stock/{ticker}/spot-exposures/strike": "GEX",
    "/api/stock/{ticker}/spot-exposures/expiry-strike": "GEX",
    "/api/stock/{ticker}/flow-per-strike-intraday": "FLOW",
    "/api/stock/{ticker}/flow-recent": "FLOW",
    "/api/stock/{ticker}/flow-per-strike": "FLOW",
    "/api/stock/{ticker}/greek-exposure": "GREEKS",
    "/api/stock/{ticker}/greek-exposure/strike": "GREEKS",
    "/api/stock/{ticker}/greek-exposure/expiry": "GREEKS",
    "/api/stock/{ticker}/ohlc/{candle_size}": "PRICE",
    "/api/stock/{ticker}/oi-per-strike": "OI",
    "/api/stock/{ticker}/oi-change": "OI",
    "/api/stock/{ticker}/iv-rank": "VOL",
    "/api/stock/{ticker}/volatility/term-structure": "VOL_TERM",
    "/api/stock/{ticker}/historical-risk-reversal-skew": "VOL_SKEW",
    "/api/darkpool/{ticker}": "DARKPOOL",
    "/api/lit-flow/{ticker}": "LITFLOW"
}

PRESENCE_ONLY_ENDPOINTS = {
    "/api/stock/{ticker}/option/volume-oi-expiry",
    "/api/stock/{ticker}/option-chains",
    "/api/stock/{ticker}/option-contracts",
    "/api/market/sectors",
    "/api/market/indices",
    "/api/market/market-context",
    "/api/market/economic-calendar",
    "/api/market/top-net-impact",
    "/api/market/total-options-volume",
    "/api/stock/{ticker}/interpolated-iv",
    "/api/stock/{ticker}/volatility/realized",
    "/api/stock/{ticker}/option/stock-price-levels",
    "/api/stock/{ticker}/max-pain",
    "/api/market/market-tide",
    "/api/stock/{ticker}/flow-alerts",
    "/api/stock/{ticker}/net-prem-ticks",
    "/api/stock/{ticker}/stock-volume-price-levels"
}

def extract_all(effective_payloads: Mapping[int, Any], contexts: Mapping[int, EndpointContext]) -> Tuple[List[FeatureRow], List[LevelRow]]:
    def rank_freshness(fs: str) -> int:
        return {"FRESH": 1, "STALE_CARRY": 2, "EMPTY_VALID": 3, "ERROR": 4}.get(fs, 5)

    candidates: List[FeatureCandidate] = []
    l_rows: List[LevelRow] = []

    def _safe_extract(
        routing_key: str,
        extractor_fn: Any,
        payload: Any,
        ctx: EndpointContext,
        meta_bucket: str,
        expected_features: List[str],
    ) -> FeatureBundle:
        """Contain any unexpected extractor exceptions so live ingest never hard-crashes.

        If an extractor raises, we return a FeatureBundle with the expected feature keys set to None
        and attach a structured NA meta payload so downstream gates can block safely."""
        try:
            return extractor_fn(payload, ctx)
        except Exception as exc:
            # Log with stack trace for diagnosis but keep the engine running.
            logger.exception(
                "Extractor exception contained",
                extra={
                    "event": "extractor_exception_contained",
                    "routing_key": routing_key,
                    "extractor": getattr(extractor_fn, "__name__", str(extractor_fn)),
                    "endpoint_id": getattr(ctx, "endpoint_id", None),
                    "path": getattr(ctx, "path", None),
                    "exc_type": type(exc).__name__,
                },
            )

            lineage = {
                "metric_name": expected_features[0] if expected_features else routing_key,
                "fields_used": [],
            }
            meta = _build_error_meta(
                ctx,
                getattr(extractor_fn, "__name__", routing_key),
                lineage,
                na_reason="extractor_exception",
                details={"exc_type": type(exc).__name__, "exc_msg": str(exc)},
            )
            return FeatureBundle(
                features={k: None for k in expected_features},
                meta={meta_bucket: meta},
            )


    for eid, ctx in contexts.items():
        payload = effective_payloads.get(eid)
        routing_key = EXTRACTOR_REGISTRY.get(ctx.path)
        
        safe_stale_age = ctx.stale_age_min if ctx.stale_age_min is not None else 999999
        
        if routing_key == "GEX":
            f_bundle = _safe_extract("GEX", extract_gex_sign, payload, ctx, meta_bucket="gex", expected_features=["net_gex_sign"])
            for k, v in f_bundle.features.items():
                candidates.append(FeatureCandidate(k, v, copy.deepcopy(f_bundle.meta.get("gex", {})), rank_freshness(ctx.freshness_state), safe_stale_age, PATH_PRIORITY.get(ctx.path, 99), eid, v is None))
            
            if ctx.freshness_state not in ("ERROR", "EMPTY_VALID") and payload and grab_list(payload):
                levels = build_gex_levels(payload)
                for l_type, price, mag, details in levels:
                    meta = _build_meta(ctx, "build_gex_levels", {"metric_name": "gex_levels", "fields_used": ["strike", "gamma_exposure"]}, details)
                    meta = _annotate_level_usage_contract(meta, l_type)
                    l_rows.append({"level_type": l_type, "price": price, "magnitude": mag, "meta_json": meta})
                
        elif routing_key == "FLOW":
            f_bundle = _safe_extract("FLOW", extract_smart_whale_pressure, payload, ctx, meta_bucket="flow", expected_features=["smart_whale_pressure"])
            for k, v in f_bundle.features.items():
                candidates.append(FeatureCandidate(k, v, copy.deepcopy(f_bundle.meta.get("flow", {})), rank_freshness(ctx.freshness_state), safe_stale_age, PATH_PRIORITY.get(ctx.path, 99), eid, v is None))
                
        elif routing_key == "GREEKS":
            f_bundle = _safe_extract("GREEKS", extract_dealer_greeks, payload, ctx, meta_bucket="greeks", expected_features=["dealer_vanna", "dealer_charm", "net_gamma_exposure_notional"])
            for k, v in f_bundle.features.items():
                candidates.append(FeatureCandidate(k, v, copy.deepcopy(f_bundle.meta.get("greeks", {})), rank_freshness(ctx.freshness_state), safe_stale_age, PATH_PRIORITY.get(ctx.path, 99), eid, v is None))
                
        elif routing_key == "PRICE":
            f_bundle = _safe_extract("PRICE", extract_price_features, payload, ctx, meta_bucket="price", expected_features=["spot"])
            for k, v in f_bundle.features.items():
                candidates.append(FeatureCandidate(k, v, copy.deepcopy(f_bundle.meta.get("price", {})), rank_freshness(ctx.freshness_state), safe_stale_age, PATH_PRIORITY.get(ctx.path, 99), eid, v is None))
                
        elif routing_key == "OI":
            f_bundle = _safe_extract("OI", extract_oi_features, payload, ctx, meta_bucket="oi", expected_features=["oi_pressure"])
            for k, v in f_bundle.features.items():
                candidates.append(FeatureCandidate(k, v, copy.deepcopy(f_bundle.meta.get("oi", {})), rank_freshness(ctx.freshness_state), safe_stale_age, PATH_PRIORITY.get(ctx.path, 99), eid, v is None))
            if ctx.freshness_state not in ("ERROR", "EMPTY_VALID") and payload and grab_list(payload):
                levels = build_oi_walls(payload)
                for l_type, price, mag, details in levels:
                    meta = _build_meta(ctx, "build_oi_walls", {"metric_name": "oi_walls", "fields_used": ["strike", "open_interest"]}, details)
                    meta = _annotate_level_usage_contract(meta, l_type)
                    l_rows.append({"level_type": l_type, "price": price, "magnitude": mag, "meta_json": meta})
                
        elif routing_key == "VOL":
            f_bundle = _safe_extract("VOL", extract_volatility_features, payload, ctx, meta_bucket="vol", expected_features=["iv_rank"])
            for k, v in f_bundle.features.items():
                candidates.append(FeatureCandidate(k, v, copy.deepcopy(f_bundle.meta.get("vol", {})), rank_freshness(ctx.freshness_state), safe_stale_age, PATH_PRIORITY.get(ctx.path, 99), eid, v is None))

        elif routing_key == "VOL_TERM":
            f_bundle = _safe_extract("VOL_TERM", extract_vol_term_structure, payload, ctx, meta_bucket="vol_ts", expected_features=["vol_term_slope"])
            for k, v in f_bundle.features.items():
                candidates.append(FeatureCandidate(k, v, copy.deepcopy(f_bundle.meta.get("vol_ts", {})), rank_freshness(ctx.freshness_state), safe_stale_age, PATH_PRIORITY.get(ctx.path, 99), eid, v is None))

        elif routing_key == "VOL_SKEW":
            f_bundle = _safe_extract("VOL_SKEW", extract_vol_skew, payload, ctx, meta_bucket="skew", expected_features=["vol_skew"])
            for k, v in f_bundle.features.items():
                candidates.append(FeatureCandidate(k, v, copy.deepcopy(f_bundle.meta.get("skew", {})), rank_freshness(ctx.freshness_state), safe_stale_age, PATH_PRIORITY.get(ctx.path, 99), eid, v is None))

        elif routing_key == "DARKPOOL":
            f_bundle = _safe_extract("DARKPOOL", extract_darkpool_pressure, payload, ctx, meta_bucket="darkpool", expected_features=["darkpool_pressure"])
            for k, v in f_bundle.features.items():
                candidates.append(FeatureCandidate(k, v, copy.deepcopy(f_bundle.meta.get("darkpool", {})), rank_freshness(ctx.freshness_state), safe_stale_age, PATH_PRIORITY.get(ctx.path, 99), eid, v is None))
            if ctx.freshness_state not in ("ERROR", "EMPTY_VALID") and payload and grab_list(payload):
                levels = build_darkpool_levels(payload)
                for l_type, price, mag, details in levels:
                    meta = _build_meta(ctx, "build_darkpool_levels", {"metric_name": "darkpool_levels", "fields_used": ["price", "volume"]}, details)
                    meta = _annotate_level_usage_contract(meta, l_type)
                    l_rows.append({"level_type": l_type, "price": price, "magnitude": mag, "meta_json": meta})

        elif routing_key == "LITFLOW":
            f_bundle = _safe_extract("LITFLOW", extract_litflow_pressure, payload, ctx, meta_bucket="litflow", expected_features=["litflow_pressure"])
            for k, v in f_bundle.features.items():
                candidates.append(FeatureCandidate(k, v, copy.deepcopy(f_bundle.meta.get("litflow", {})), rank_freshness(ctx.freshness_state), safe_stale_age, PATH_PRIORITY.get(ctx.path, 99), eid, v is None))
                
        elif ctx.path not in PRESENCE_ONLY_ENDPOINTS:
            raise RuntimeError(f"CRITICAL EXTRACTOR COVERAGE GAP: Endpoint path '{ctx.path}' is not mapped in EXTRACTOR_REGISTRY and not whitelisted in PRESENCE_ONLY_ENDPOINTS.")

    grouped: Dict[str, List[FeatureCandidate]] = {}
    for c in candidates: 
        grouped.setdefault(c.feature_key, []).append(c)

    f_rows: List[FeatureRow] = []
    for f_key, group in grouped.items():
        group.sort(key=lambda x: (x.is_none, x.freshness_rank, x.stale_age, x.path_priority, x.endpoint_id))
        best = group[0]
        
        for other in group[1:]:
            if (other.is_none == best.is_none and other.freshness_rank == best.freshness_rank and 
                other.stale_age == best.stale_age and other.path_priority == best.path_priority):
                if best.feature_value is not None and other.feature_value is not None:
                    if not math.isclose(best.feature_value, other.feature_value, abs_tol=1e-9):
                        raise RuntimeError(f"FEATURE_CONFLICT:{f_key} - Endpoint {best.endpoint_id} vs {other.endpoint_id} generated divergent values at equal rank.")
                    
        meta = copy.deepcopy(best.meta_json)
        if len(group) > 1:
            meta.setdefault("details", {})["shadowed_candidates"] = [
                {"endpoint_id": c.endpoint_id, "is_none": c.is_none, "freshness_rank": c.freshness_rank} for c in group[1:]
            ]
            
        f_rows.append({"feature_key": best.feature_key, "feature_value": best.feature_value, "meta_json": meta})
        
        metric_family = meta.get("metric_lineage", {}).get("metric_name", "unknown")
        if best.feature_value is not None and math.isfinite(best.feature_value):
            logger.info(
                f"Feature emitted: {f_key}", 
                extra={"counter": "features_emitted_by_family", "family": metric_family, "feature_key": f_key}
            )
        else:
            logger.warning(
                f"Feature suppressed: {f_key}", 
                extra={"counter": "features_suppressed_by_family", "family": metric_family, "feature_key": f_key}
            )

    return f_rows, l_rows

# Fixed: Changed "module" to "module_name" to prevent LogRecord KeyErrors
logger.info("Features module initialized successfully", extra={"event": "module_init", "module_name": "features"})