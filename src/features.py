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

                best_row_date: Optional[datetime.datetime] = None
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

        best_row_date: Optional[datetime.datetime] = None
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


def _normalize_effective_time_source(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    if not normalized:
        return None
    return normalized


def _normalize_output_domain(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _with_output_domain(
    lineage: Dict[str, Any],
    *,
    emitted_units: Optional[str] = None,
    raw_input_units: Optional[str] = None,
    bounded_output: Optional[bool] = None,
    expected_bounds: Any = None,
    output_domain: str = "unbounded_scalar",
) -> Dict[str, Any]:
    enriched = dict(lineage)
    if emitted_units is not None:
        enriched["emitted_units"] = emitted_units
    if raw_input_units is not None:
        enriched["raw_input_units"] = raw_input_units
    if bounded_output is not None:
        enriched["bounded_output"] = bool(bounded_output)
    normalized_bounds = _normalize_expected_bounds(expected_bounds)
    if normalized_bounds is not None:
        enriched["expected_bounds"] = normalized_bounds
    enriched["output_domain"] = output_domain
    enriched["output_domain_contract_version"] = OUTPUT_DOMAIN_CONTRACT_VERSION
    return enriched


def _feature_use_role_for_path(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    override = LEGACY_PATH_USE_ROLE_OVERRIDES.get(path)
    if override:
        return override
    return None


def _feature_use_role_details(role: str) -> Dict[str, Any]:
    base = FEATURE_USE_ROLE_DEFAULTS.get(role)
    if base is None:
        base = FEATURE_USE_ROLE_DEFAULTS["signal-critical"]
        role = "signal-critical"
    return {"feature_use_role": role, **base}


def _resolve_feature_use_role(
    meta_json: Dict[str, Any],
    *,
    path: Optional[str] = None,
    role_override: Optional[str] = None,
) -> Dict[str, Any]:
    explicit_role = role_override
    if explicit_role is None:
        explicit_role = meta_json.get("feature_use_role")
    if explicit_role is None:
        explicit_role = _feature_use_role_for_path(path)
    if explicit_role is None:
        explicit_role = "signal-critical"

    details = _feature_use_role_details(str(explicit_role))
    meta_json["feature_use_role"] = details["feature_use_role"]
    meta_json["decision_path"] = bool(details["decision_path"])
    meta_json["missing_affects_confidence"] = bool(details["missing_affects_confidence"])
    meta_json["stale_affects_confidence"] = bool(details["stale_affects_confidence"])
    meta_json["feature_use_contract_version"] = FEATURE_USE_CONTRACT_VERSION
    return details


def _propagate_metric_lineage(meta: Dict[str, Any], ctx: EndpointContext) -> None:
    lineage = meta.get("metric_lineage") if isinstance(meta.get("metric_lineage"), dict) else {}
    effective_ts = meta.get("effective_ts_utc")
    if effective_ts is None:
        effective_ts = meta.get("effective_at_utc")
    if effective_ts is None:
        effective_ts = getattr(ctx, "effective_timestamp_utc", None)
    if effective_ts is not None:
        lineage["effective_ts_utc"] = effective_ts

    if "event_time_utc" not in lineage and meta.get("event_time_utc") is not None:
        lineage["event_time_utc"] = meta.get("event_time_utc")
    if "publish_time_utc" not in lineage and meta.get("publish_time_utc") is not None:
        lineage["publish_time_utc"] = meta.get("publish_time_utc")
    if "source_revision" not in lineage and meta.get("source_revision") is not None:
        lineage["source_revision"] = meta.get("source_revision")
    if "effective_time_source" not in lineage:
        time_source = _normalize_effective_time_source(meta.get("effective_time_source"))
        if time_source is not None:
            lineage["effective_time_source"] = time_source
    if "time_provenance_degraded" not in lineage:
        lineage["time_provenance_degraded"] = bool(meta.get("time_provenance_degraded", False))

    for key in (
        "emitted_units",
        "raw_input_units",
        "bounded_output",
        "expected_bounds",
        "output_domain",
        "output_domain_contract_version",
    ):
        if key not in lineage and meta.get(key) is not None:
            lineage[key] = meta.get(key)

    meta["metric_lineage"] = lineage


def _extract_payload_provenance(payload: Any) -> Dict[str, Any]:
    """Return machine-readable timing provenance from raw payload where possible."""

    prov: Dict[str, Any] = {
        "event_time_utc": None,
        "publish_time_utc": None,
        "effective_ts_utc": None,
        "effective_time_source": "unknown",
        "source_revision": None,
        "time_provenance_degraded": False,
    }

    candidates: List[Mapping[str, Any]] = []
    if isinstance(payload, Mapping):
        candidates.append(payload)
        rows = payload.get("data")
        if isinstance(rows, list) and rows and isinstance(rows[0], Mapping):
            candidates.append(rows[0])
    elif isinstance(payload, list) and payload and isinstance(payload[0], Mapping):
        candidates.append(payload[0])

    latest_publish: Optional[datetime.datetime] = None
    latest_event: Optional[datetime.datetime] = None
    latest_effective: Optional[datetime.datetime] = None

    for obj in candidates:
        if not isinstance(obj, Mapping):
            continue

        for key in PUBLISH_TIME_KEYS:
            ts = _parse_utc_dt(obj.get(key))
            if ts is not None and (latest_publish is None or ts > latest_publish):
                latest_publish = ts

        for key in EVENT_TIME_KEYS:
            ts = _parse_utc_dt(obj.get(key))
            if ts is not None and (latest_event is None or ts > latest_event):
                latest_event = ts

        for key in EFFECTIVE_TIME_KEYS:
            ts = _parse_utc_dt(obj.get(key))
            if ts is not None and (latest_effective is None or ts > latest_effective):
                latest_effective = ts

        for key in SOURCE_REVISION_KEYS:
            if obj.get(key) is not None:
                prov["source_revision"] = obj.get(key)
                break

    # Canonical precedence:
    # 1) explicit effective timestamp from payload
    # 2) event timestamp
    # 3) publish/report timestamp
    eff = latest_effective
    effective_source = "payload_effective" if eff is not None else None
    degraded = False

    if eff is None and latest_event is not None:
        eff = latest_event
        effective_source = "event_time_fallback"

    if eff is None and latest_publish is not None:
        eff = latest_publish
        effective_source = "publish_time_fallback"
        degraded = True

    # Date-only snapshot fallback: deterministic midnight UTC of provided date.
    if eff is None:
        inferred = _infer_daily_snapshot_effective_ts_utc(payload)
        if inferred is not None:
            eff = inferred
            effective_source = "date_only_snapshot_fallback"
            degraded = True

    prov["event_time_utc"] = latest_event.isoformat() if latest_event is not None else None
    prov["publish_time_utc"] = latest_publish.isoformat() if latest_publish is not None else None
    prov["effective_ts_utc"] = eff.isoformat() if eff is not None else None
    prov["effective_time_source"] = effective_source or "unknown"
    prov["time_provenance_degraded"] = degraded
    return prov


def _emit_endpoint_payload_observability(
    *,
    ctx: EndpointContext,
    payload: Any,
    feature_output_reason: str,
    parsed_row_count: int,
    rows_discarded_before_extraction: int = 0,
    effective_timestamp_source: str = "unknown",
) -> None:
    """Emit machine-readable payload observability when a feature cannot be produced safely."""

    if not getattr(ctx, "logger", None):
        return
    try:
        ctx.logger.info(
            "endpoint_payload_observability",
            extra={
                "payload_type": type(payload).__name__,
                "feature_output_reason": feature_output_reason,
                "parsed_row_count": int(parsed_row_count),
                "rows_discarded_before_extraction": int(rows_discarded_before_extraction),
                "effective_timestamp_source": effective_timestamp_source,
            },
        )
    except Exception:
        # Observability must never break extraction.
        return


def _coerce_ts_iso(value: Any) -> Optional[str]:
    """Coerce epoch-like or datetime-like values to UTC ISO-8601.

    Accepts:
    - datetime (naive assumed UTC)
    - int/float epoch seconds
    - int/float epoch milliseconds when magnitude is too large for seconds
    - ISO8601-ish strings accepted by _parse_utc_dt
    """
    if value is None:
        return None
    try:
        if isinstance(value, datetime.datetime):
            dt = value
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=datetime.timezone.utc)
            else:
                dt = dt.astimezone(datetime.timezone.utc)
            return dt.isoformat()
        if isinstance(value, (int, float)):
            x = float(value)
            # Heuristic: values far beyond plausible epoch-seconds are epoch-millis.
            # 1e12 ~= 2001-09-09 in milliseconds and ~33658 CE in seconds.
            if abs(x) >= 1e12:
                x = x / 1000.0
            dt = datetime.datetime.fromtimestamp(x, tz=datetime.timezone.utc)
            return dt.isoformat()
        if isinstance(value, str):
            dt = _parse_utc_dt(value)
            return dt.isoformat() if dt is not None else None
    except Exception:
        return None
    return None


def _parse_strict_ts(row: Dict[str, Any], key: str) -> Optional[float]:
    """Parse a timestamp field to epoch seconds with strict validation.

    Returns:
      - float epoch seconds on success
      - None when field missing/unparseable
    Normalizes epoch-millis to seconds when detected.
    """
    if row.get(key) is None:
        return None
    v = row.get(key)
    try:
        if isinstance(v, datetime.datetime):
            dt = v
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=datetime.timezone.utc)
            else:
                dt = dt.astimezone(datetime.timezone.utc)
            return float(dt.timestamp())
        if isinstance(v, (int, float)):
            x = float(v)
            if abs(x) >= 1e12:
                x = x / 1000.0
            # Reject absurdly out-of-range years by attempting conversion.
            datetime.datetime.fromtimestamp(x, tz=datetime.timezone.utc)
            return x
        if isinstance(v, str):
            s = v.strip()
            if not s:
                return None
            # Numeric strings
            try:
                x = float(s)
                if abs(x) >= 1e12:
                    x = x / 1000.0
                datetime.datetime.fromtimestamp(x, tz=datetime.timezone.utc)
                return x
            except Exception:
                pass
            dt = _parse_utc_dt(s)
            return float(dt.timestamp()) if dt is not None else None
    except Exception:
        return None
    return None


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
    error_meta = _build_error_meta(
        ctx,
        extractor_name,
        lineage,
        "contract_normalization_invalid",
        details={
            **_extract_payload_provenance(payload),
            "contract_normalization": summary.as_dict() if hasattr(summary, "as_dict") else {},
        },
    )
    features = {key: None for key in feature_keys}
    return FeatureBundle(features, {meta_key: error_meta})


def _normalized_identity_for_row(contract_map: Mapping[int, Any], idx: int) -> Optional[Any]:
    return contract_map.get(idx)


def _build_feature_meta(ctx: EndpointContext, extractor_name: str, lineage: Dict[str, Any], details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    meta = _build_meta(ctx, extractor_name, lineage, details)
    _resolve_feature_use_role(meta, path=getattr(ctx, "path", None))
    _propagate_metric_lineage(meta, ctx)
    return meta


def _build_error_feature_meta(
    ctx: EndpointContext,
    extractor_name: str,
    lineage: Dict[str, Any],
    reason: str,
    *,
    details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    meta = _build_error_meta(ctx, extractor_name, lineage, reason, details=details)
    _resolve_feature_use_role(meta, path=getattr(ctx, "path", None))
    _propagate_metric_lineage(meta, ctx)
    return meta


def _build_meta(ctx: EndpointContext, extractor_name: str, lineage: Dict[str, Any], details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    meta = {
        "extractor": extractor_name,
        "freshness_state": ctx.freshness_state,
        "stale_age_min": ctx.stale_age_min,
        "na_reason": ctx.na_reason,
        "source_endpoints": [{"path": ctx.path, "method": ctx.method}],
        "metric_lineage": copy.deepcopy(lineage),
    }
    if details:
        meta.update(details)

    if "effective_ts_utc" not in meta:
        if meta.get("effective_at_utc") is not None:
            meta["effective_ts_utc"] = meta.get("effective_at_utc")
        elif getattr(ctx, "effective_timestamp_utc", None) is not None:
            meta["effective_ts_utc"] = ctx.effective_timestamp_utc
    meta["effective_time_source"] = _normalize_effective_time_source(meta.get("effective_time_source")) or "unknown"
    meta["output_domain"] = _normalize_output_domain(meta.get("output_domain"))
    if meta.get("output_domain") is None:
        meta["output_domain"] = "unbounded_scalar"
    meta["output_domain_contract_version"] = meta.get("output_domain_contract_version") or OUTPUT_DOMAIN_CONTRACT_VERSION
    if "emitted_units" not in meta:
        meta["emitted_units"] = lineage.get("emitted_units")
    if "raw_input_units" not in meta:
        meta["raw_input_units"] = lineage.get("raw_input_units")
    if "bounded_output" not in meta and lineage.get("bounded_output") is not None:
        meta["bounded_output"] = bool(lineage.get("bounded_output"))
    if "expected_bounds" not in meta:
        normalized_bounds = _normalize_expected_bounds(lineage.get("expected_bounds"))
        if normalized_bounds is not None:
            meta["expected_bounds"] = normalized_bounds

    metric_lineage = meta.get("metric_lineage")
    if isinstance(metric_lineage, dict):
        if "effective_ts_utc" not in metric_lineage and meta.get("effective_ts_utc") is not None:
            metric_lineage["effective_ts_utc"] = meta.get("effective_ts_utc")
        if "effective_time_source" not in metric_lineage:
            metric_lineage["effective_time_source"] = meta.get("effective_time_source")
        if "time_provenance_degraded" not in metric_lineage:
            metric_lineage["time_provenance_degraded"] = bool(meta.get("time_provenance_degraded", False))
        for key in (
            "emitted_units",
            "raw_input_units",
            "bounded_output",
            "expected_bounds",
            "output_domain",
            "output_domain_contract_version",
        ):
            if key not in metric_lineage and meta.get(key) is not None:
                metric_lineage[key] = meta.get(key)

    return meta


def _build_error_meta(
    ctx: EndpointContext,
    extractor_name: str,
    lineage: Dict[str, Any],
    reason: str,
    details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    meta = _build_meta(ctx, extractor_name, lineage, details)
    meta["na_reason"] = reason
    return meta


def _normalize_signed(x: Optional[float], scale: float = 1.0) -> Optional[float]:
    v = safe_float(x)
    if v is None or scale <= 0:
        return None
    return max(min(v / scale, 1.0), -1.0)


def _confidence_degraded_meta(
    ctx: EndpointContext,
    extractor_name: str,
    lineage: Dict[str, Any],
    details: Dict[str, Any],
) -> Dict[str, Any]:
    meta = _build_meta(ctx, extractor_name, lineage, details)
    meta["confidence_impact"] = "DEGRADED"
    return meta


def _iter_rows(payload: Any) -> Iterable[Dict[str, Any]]:
    for r in _grab_list(payload):
        if isinstance(r, dict):
            yield r


def _sum_key(payload: Any, keys: Iterable[str]) -> Optional[float]:
    total = 0.0
    any_seen = False
    for row in _iter_rows(payload):
        for k in keys:
            v = _as_float(row.get(k))
            if v is not None:
                total += v
                any_seen = True
                break
    return total if any_seen else None


def _first_non_null(row: Dict[str, Any], keys: Iterable[str]) -> Any:
    for k in keys:
        v = row.get(k)
        if v is not None:
            return v
    return None


def _safe_ratio(num: float, den: float) -> Optional[float]:
    if den == 0:
        return None
    return num / den


def _to_utc_iso(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, datetime.datetime):
        dt = value
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.timezone.utc)
        else:
            dt = dt.astimezone(datetime.timezone.utc)
        return dt.isoformat()
    if isinstance(value, datetime.date):
        dt = datetime.datetime(value.year, value.month, value.day, tzinfo=datetime.timezone.utc)
        return dt.isoformat()
    if isinstance(value, (int, float)):
        try:
            return datetime.datetime.fromtimestamp(float(value), tz=datetime.timezone.utc).isoformat()
        except Exception:
            return None
    if isinstance(value, str):
        dt = _parse_utc_dt(value)
        return dt.isoformat() if dt is not None else None
    return None


def _extract_spot_reference(payload: Any) -> Optional[float]:
    if isinstance(payload, dict):
        for k in SPOT_KEYS:
            v = safe_float(payload.get(k))
            if v is not None:
                return v
        rows = grab_list(payload)
    else:
        rows = grab_list(payload)

    for row in rows:
        if not isinstance(row, dict):
            continue
        for k in SPOT_KEYS:
            v = safe_float(row.get(k))
            if v is not None:
                return v
    return None


def _to_iso_or_none(value: Any) -> Optional[str]:
    return _to_utc_iso(value)


def _feature_row(
    key: str,
    val: Optional[float],
    meta: Dict[str, Any],
    *,
    role_override: Optional[str] = None,
) -> FeatureRow:
    if meta is None:
        meta = {}
    _resolve_feature_use_role(meta, path=meta.get("source_endpoints", [{}])[0].get("path") if isinstance(meta.get("source_endpoints"), list) and meta.get("source_endpoints") else None, role_override=role_override)
    return {
        "feature_key": key,
        "feature_value": val,
        "meta_json": meta,
    }


def _level_row(level_type: str, price: Optional[float], magnitude: Optional[float], meta: Dict[str, Any]) -> LevelRow:
    return {"level_type": level_type, "price": price, "magnitude": magnitude, "meta_json": meta}


def _extract_interval_minutes(path: str, payload: Any) -> Optional[int]:
    if "ohlc" in path:
        if isinstance(payload, dict):
            for key in ("interval", "timeframe", "candle_size"):
                raw = payload.get(key)
                if isinstance(raw, str):
                    m = re.match(r"^(\d+)\s*(m|min|minute|minutes)$", raw.strip(), flags=re.I)
                    if m:
                        return int(m.group(1))
                    m = re.match(r"^(\d+)$", raw.strip())
                    if m:
                        return int(m.group(1))
                elif isinstance(raw, (int, float)):
                    return int(raw)
        m = re.search(r"/ohlc/(\d+)", path)
        if m:
            return int(m.group(1))
        return 1
    return None


def _ctx_sort_keys(ctx: EndpointContext, payload: Any) -> Tuple[int, int, int, int]:
    freshness_rank = 0 if ctx.freshness_state == "FRESH" else 1 if ctx.freshness_state == "EMPTY_VALID" else 2 if ctx.freshness_state == "STALE_CARRY" else 3
    stale_age = int(ctx.stale_age_min if ctx.stale_age_min is not None else 999999)
    path_priority = PATH_PRIORITY.get(ctx.path, 999)
    endpoint_id = id(payload)
    return freshness_rank, stale_age, path_priority, endpoint_id


def extract_price_features(ohlc_payload: Any, ctx: EndpointContext) -> FeatureBundle:
    lineage = _with_output_domain({
        "metric_name": "spot",
        "fields_used": ["close", "end_time|start_time|t|timestamp|time|datetime|date"],
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
        "quality_policy": "None on filtered zeros / unparseable rows to avoid false baseline certainty; neutral only for structurally empty valid payload; suppress on contract normalization failure",
        "criticality": "CRITICAL"
    }, emitted_units="normalized_directional_pressure", raw_input_units="Net Premium Flow (USD)", bounded_output=True, expected_bounds=(-1.0, 1.0), output_domain="closed_interval")

    if is_na(flow_payload) or ctx.freshness_state == "ERROR":
        return FeatureBundle({"smart_whale_pressure": None}, {"flow": _build_error_meta(ctx, "extract_smart_whale_pressure", lineage, ctx.na_reason or "missing_dependency")})

    raw_rows = None
    if isinstance(flow_payload, list):
        raw_rows = flow_payload
    elif isinstance(flow_payload, dict):
        for key in ("data", "trades", "results", "result", "history", "items"):
            if key in flow_payload:
                raw_rows = flow_payload.get(key)
                break

    if isinstance(raw_rows, list) and raw_rows and not all(isinstance(t, dict) for t in raw_rows):
        _emit_endpoint_payload_observability(
            ctx=ctx,
            payload=flow_payload,
            feature_output_reason="schema_non_dict_rows",
            parsed_row_count=0,
            rows_discarded_before_extraction=len(raw_rows),
            effective_timestamp_source="unknown",
        )
        return FeatureBundle({"smart_whale_pressure": None}, {"flow": _build_error_meta(ctx, "extract_smart_whale_pressure", lineage, "schema_non_dict_rows")})

    trades = grab_list(flow_payload)

    # Only structurally empty payloads get a neutral output.
    # Payloads with rows that are present but semantically unusable are handled below as NA.
    if not trades:
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
        _emit_endpoint_payload_observability(
            ctx=ctx,
            payload=flow_payload,
            feature_output_reason="contract_normalization_invalid",
            parsed_row_count=0,
            rows_discarded_before_extraction=len(trades),
            effective_timestamp_source="unknown",
        )
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

    whale_call = 0.0
    whale_put = 0.0
    valid_count = 0
    skip_missing_fields = 0
    skip_bad_type = 0
    skip_bad_side = 0
    skip_threshold = 0

    pc_map = {"C": "CALL", "CALL": "CALL", "CALLS": "CALL", "P": "PUT", "PUT": "PUT", "PUTS": "PUT"}
    side_map = {"ASK": "BULL", "BUY": "BULL", "BULLISH": "BULL", "BOT": "BULL", "BID": "BEAR", "SELL": "BEAR", "BEARISH": "BEAR", "SOLD": "BEAR"}

    for idx, t in enumerate(trades):
        prem = safe_float(t.get("premium") or t.get("price") or t.get("notional"))
        dte = safe_float(t.get("dte") or t.get("days_to_expiration"))
        side_raw = t.get("side") or t.get("direction")
        identity = _normalized_identity_for_row(contract_map, idx)
        pc_raw = identity.put_call if identity is not None else (
            t.get("put_call") or t.get("option_type") or t.get("type") or t.get("pc")
        )

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

    # filtered/unparseable/unknown-side => NA, not neutral
    if valid_count == 0:
        reason = "filtered_to_zero"
        if skip_bad_side > 0 and skip_bad_type == 0 and skip_missing_fields == 0 and skip_threshold == 0:
            reason = "unknown_side_labels"
        elif skip_threshold == 0 and (skip_missing_fields > 0 or skip_bad_type > 0):
            reason = "unparseable_rows"

        _emit_endpoint_payload_observability(
            ctx=ctx,
            payload=flow_payload,
            feature_output_reason=reason,
            parsed_row_count=0,
            rows_discarded_before_extraction=len(trades),
            effective_timestamp_source="unknown",
        )
        return FeatureBundle(
            {"smart_whale_pressure": None},
            {
                "flow": _build_error_meta(
                    ctx,
                    "extract_smart_whale_pressure",
                    lineage,
                    reason,
                    details={
                        "n_trades": len(trades),
                        "skip_missing_fields": skip_missing_fields,
                        "skip_bad_type": skip_bad_type,
                        "skip_bad_side": skip_bad_side,
                        "skip_threshold": skip_threshold,
                    },
                )
            },
        )

    net = whale_call - whale_put
    details = {
        **_extract_payload_provenance(flow_payload),
        "contract_normalization": norm_summary.as_dict(),
        "n_raw_trades": len(trades),
        "n_valid": valid_count,
        "net_prem": net,
        "status": "ok",
    }
    meta = _build_meta(ctx, "extract_smart_whale_pressure", lineage, details)
    return FeatureBundle({"smart_whale_pressure": _normalize_signed(net, scale=norm_scale)}, {"flow": meta})

def extract_dealer_greeks(greek_payload: Any, ctx: EndpointContext, norm_scale: float = 1_000_000_000.0) -> FeatureBundle:
    keys = ["dealer_vanna", "dealer_charm", "net_gamma_exposure_notional"]
    lineage = _with_output_domain({
        "metric_name": "dealer_greeks_bundle",
        "fields_used": ["vanna", "charm", "gamma_exposure", "gamma", "notional"],
        "units_expected": "Normalized signed exposure [-1,1] with raw notionals preserved in meta",
        "normalization": f"normalize_signed by {norm_scale}; parse aggregate greek exposure endpoints",
        "session_applicability": "RTH",
        "quality_policy": "None if endpoint missing or schema invalid",
        "criticality": "CRITICAL"
    }, emitted_units="normalized_signed_exposure", raw_input_units="Notional Exposure (USD)", bounded_output=True, expected_bounds=(-1.0, 1.0), output_domain="closed_interval")

    if is_na(greek_payload) or ctx.freshness_state == "ERROR":
        error_meta = _build_error_meta(ctx, "extract_dealer_greeks", lineage, ctx.na_reason or "missing_dependency")
        return FeatureBundle({k: None for k in keys}, {"greeks": error_meta})

    rows = grab_list(greek_payload)
    if rows:
        row = rows[0]
        vanna = safe_float(row.get("vanna") or row.get("dealer_vanna"))
        charm = safe_float(row.get("charm") or row.get("dealer_charm"))
        gamma = safe_float(
            row.get("gamma_exposure")
            or row.get("gamma")
            or row.get("gex")
            or row.get("gammaNotional")
            or row.get("net_gamma_exposure_notional")
            or row.get("notional")
        )
    elif isinstance(greek_payload, dict):
        vanna = safe_float(greek_payload.get("vanna") or greek_payload.get("dealer_vanna"))
        charm = safe_float(greek_payload.get("charm") or greek_payload.get("dealer_charm"))
        gamma = safe_float(
            greek_payload.get("gamma_exposure")
            or greek_payload.get("gamma")
            or greek_payload.get("gex")
            or greek_payload.get("gammaNotional")
            or greek_payload.get("net_gamma_exposure_notional")
            or greek_payload.get("notional")
        )
    else:
        vanna = charm = gamma = None

    prov = _extract_payload_provenance(greek_payload)
    meta = _build_meta(ctx, "extract_dealer_greeks", lineage, {
        **prov,
        "vanna_raw": vanna,
        "charm_raw": charm,
        "gamma_raw": gamma,
        "status": "ok" if any(v is not None for v in (vanna, charm, gamma)) else "partial",
    })

    return FeatureBundle({
        "dealer_vanna": _normalize_signed(vanna, scale=norm_scale),
        "dealer_charm": _normalize_signed(charm, scale=norm_scale),
        "net_gamma_exposure_notional": _normalize_signed(gamma, scale=norm_scale),
    }, {"greeks": meta})


def extract_gex_sign(gex_payload: Any, ctx: EndpointContext, min_abs_notional: float = 1_000_000.0) -> FeatureBundle:
    lineage = {
        "metric_name": "net_gex_sign",
        "fields_used": ["gamma_exposure", "gex", "notional"],
        "units_expected": "-1 / 0 / +1 sign of net gamma exposure",
        "normalization": "sign of aggregate gamma exposure; 0 if |exposure| < threshold",
        "session_applicability": "RTH",
        "quality_policy": "None if aggregate gamma exposure cannot be inferred safely",
        "criticality": "CRITICAL",
        "emitted_units": "sign",
        "raw_input_units": "Notional Exposure (USD)",
        "bounded_output": True,
        "expected_bounds": {"lower": -1.0, "upper": 1.0, "inclusive": True},
    }
    if is_na(gex_payload) or ctx.freshness_state == "ERROR":
        return FeatureBundle({"net_gex_sign": None}, {"gex": _build_error_meta(ctx, "extract_gex_sign", lineage, ctx.na_reason or "missing_dependency")})

    rows = grab_list(gex_payload)
    if rows:
        total = 0.0
        any_seen = False
        for row in rows:
            if not isinstance(row, dict):
                continue
            candidates = (
                row.get("gamma_exposure"),
                row.get("gex"),
                row.get("gamma"),
                row.get("notional"),
                row.get("value"),
                row.get("net_gex"),
                row.get("gamma_exposure_notional"),
                row.get("net_gamma_exposure_notional"),
                row.get("total_gamma_exposure"),
            )
            val = next((safe_float(v) for v in candidates if safe_float(v) is not None), None)
            if val is None:
                continue
            total += val
            any_seen = True
        gex = total if any_seen else None
    elif isinstance(gex_payload, dict):
        candidates = (
            gex_payload.get("gamma_exposure"),
            gex_payload.get("gex"),
            gex_payload.get("gamma"),
            gex_payload.get("notional"),
            gex_payload.get("value"),
            gex_payload.get("net_gex"),
            gex_payload.get("gamma_exposure_notional"),
            gex_payload.get("net_gamma_exposure_notional"),
            gex_payload.get("total_gamma_exposure"),
        )
        gex = next((safe_float(v) for v in candidates if safe_float(v) is not None), None)
        # Schema-drift fallback: some vendor variants nest the aggregate under common
        # containers such as {"data": {...}} or {"result": {...}}. Only attempt a narrow
        # deterministic fallback when a direct aggregate key is absent.
        if gex is None:
            for key in ("data", "result", "results", "summary", "totals"):
                nested = gex_payload.get(key)
                if isinstance(nested, dict):
                    nested_candidates = (
                        nested.get("gamma_exposure"),
                        nested.get("gex"),
                        nested.get("gamma"),
                        nested.get("notional"),
                        nested.get("value"),
                        nested.get("net_gex"),
                        nested.get("gamma_exposure_notional"),
                        nested.get("net_gamma_exposure_notional"),
                        nested.get("total_gamma_exposure"),
                    )
                    gex = next((safe_float(v) for v in nested_candidates if safe_float(v) is not None), None)
                    if gex is not None:
                        break
    else:
        gex = None

    if gex is None:
        _emit_endpoint_payload_observability(
            ctx=ctx,
            payload=gex_payload,
            feature_output_reason="unparseable_rows",
            parsed_row_count=0,
            rows_discarded_before_extraction=len(rows) if isinstance(rows, list) else 0,
            effective_timestamp_source="unknown",
        )
        return FeatureBundle({"net_gex_sign": None}, {"gex": _build_error_meta(ctx, "extract_gex_sign", lineage, "unparseable_rows")})

    if abs(gex) < min_abs_notional:
        sign = 0.0
        status = "small_exposure_neutral"
    else:
        sign = 1.0 if gex > 0 else -1.0
        status = "ok"

    prov = _extract_payload_provenance(gex_payload)
    meta = _build_meta(ctx, "extract_gex_sign", lineage, {
        **prov,
        "gex_raw": gex,
        "min_abs_notional": min_abs_notional,
        "status": status,
    })
    return FeatureBundle({"net_gex_sign": sign}, {"gex": meta})


def extract_oi_pressure(oi_payload: Any, ctx: EndpointContext, norm_scale: float = 100_000.0) -> FeatureBundle:
    lineage = _with_output_domain({
        "metric_name": "oi_pressure",
        "fields_used": ["call_oi", "put_oi", "open_interest", "type|put_call"],
        "units_expected": "Directional imbalance ratio [-1,1]",
        "normalization": "call_minus_put over total OI",
        "session_applicability": "RTH",
        "quality_policy": "None if put/call OI not inferable safely",
        "criticality": "CRITICAL"
    }, emitted_units="directional_imbalance_ratio", raw_input_units="Open Interest (contracts)", bounded_output=True, expected_bounds=(-1.0, 1.0), output_domain="closed_interval")

    if is_na(oi_payload) or ctx.freshness_state == "ERROR":
        return FeatureBundle({"oi_pressure": None}, {"oi": _build_error_meta(ctx, "extract_oi_pressure", lineage, ctx.na_reason or "missing_dependency")})

    rows = grab_list(oi_payload)

    # Prefer direct aggregate call/put OI if available
    call_oi = put_oi = None
    if isinstance(oi_payload, dict):
        call_oi = safe_float(oi_payload.get("call_oi") or oi_payload.get("calls_oi") or oi_payload.get("callOpenInterest"))
        put_oi = safe_float(oi_payload.get("put_oi") or oi_payload.get("puts_oi") or oi_payload.get("putOpenInterest"))

    skip_missing_oi = 0
    skip_bad_type = 0

    if call_oi is None or put_oi is None:
        call_oi = 0.0
        put_oi = 0.0
        any_valid = False
        pc_map = {"C": "CALL", "CALL": "CALL", "CALLS": "CALL", "P": "PUT", "PUT": "PUT", "PUTS": "PUT"}

        norm_summary = normalize_option_rows(rows)
        contract_map = normalized_contract_map(norm_summary)

        for idx, r in enumerate(rows):
            oi = safe_float(
                r.get("open_interest")
                or r.get("oi")
                or r.get("openInterest")
                or r.get("open_interest_change")
                or r.get("oi_change")
            )
            identity = _normalized_identity_for_row(contract_map, idx)
            pc_raw = identity.put_call if identity is not None else (
                r.get("put_call") or r.get("option_type") or r.get("type") or r.get("pc")
            )
            if oi is None or is_na(pc_raw):
                skip_missing_oi += 1
                continue
            pc_norm = pc_raw if identity is not None else pc_map.get(str(pc_raw).upper().strip())
            if not pc_norm:
                skip_bad_type += 1
                continue
            if pc_norm == "CALL":
                call_oi += oi
            else:
                put_oi += oi
            any_valid = True

        if not any_valid:
            # Schema-drift fallback for payloads that provide aggregate open interest under
            # non-standard aliases (e.g. "put_open_interest"/"call_open_interest").
            if isinstance(oi_payload, dict):
                alt_call = safe_float(
                    oi_payload.get("call_open_interest")
                    or oi_payload.get("calls_open_interest")
                    or oi_payload.get("call_oi_total")
                    or oi_payload.get("total_call_oi")
                )
                alt_put = safe_float(
                    oi_payload.get("put_open_interest")
                    or oi_payload.get("puts_open_interest")
                    or oi_payload.get("put_oi_total")
                    or oi_payload.get("total_put_oi")
                )
                if alt_call is not None and alt_put is not None:
                    call_oi = alt_call
                    put_oi = alt_put
                    any_valid = True

        if not any_valid:
            reason = "unparseable_rows" if (skip_missing_oi > 0 or skip_bad_type > 0) else "filtered_to_zero"
            _emit_endpoint_payload_observability(
                ctx=ctx,
                payload=oi_payload,
                feature_output_reason=reason,
                parsed_row_count=0,
                rows_discarded_before_extraction=len(rows),
                effective_timestamp_source="unknown",
            )
            return FeatureBundle({"oi_pressure": None}, {"oi": _build_error_meta(ctx, "extract_oi_pressure", lineage, reason)})

    total = call_oi + put_oi
    if total == 0:
        # True empty/zero aggregate OI is neutral, but degraded
        prov = _extract_payload_provenance(oi_payload)
        meta = _confidence_degraded_meta(ctx, "extract_oi_pressure", lineage, {
            **prov,
            "call_oi": call_oi,
            "put_oi": put_oi,
            "status": "computed_neutral_from_empty",
        })
        meta["freshness_state"] = "EMPTY_VALID"
        meta["na_reason"] = "empty_payload"
        return FeatureBundle({"oi_pressure": 0.0}, {"oi": meta})

    ratio = (call_oi - put_oi) / total
    prov = _extract_payload_provenance(oi_payload)
    meta = _build_meta(ctx, "extract_oi_pressure", lineage, {
        **prov,
        "call_oi": call_oi,
        "put_oi": put_oi,
        "status": "ok",
    })
    return FeatureBundle({"oi_pressure": max(min(ratio, 1.0), -1.0)}, {"oi": meta})


def extract_iv_rank(iv_payload: Any, ctx: EndpointContext) -> FeatureBundle:
    lineage = _with_output_domain({
        "metric_name": "iv_rank",
        "fields_used": ["iv_rank", "rank", "percentile"],
        "units_expected": "Percentile rank [0,1]",
        "normalization": "iv_rank/100 when source appears as percentage",
        "session_applicability": "RTH",
        "quality_policy": "None if IV rank unavailable",
        "criticality": "NON_CRITICAL"
    }, emitted_units="percentile_rank", raw_input_units="Percentile Rank", bounded_output=True, expected_bounds=(0.0, 1.0), output_domain="closed_interval")

    if is_na(iv_payload) or ctx.freshness_state == "ERROR":
        return FeatureBundle({"iv_rank": None}, {"iv": _build_error_meta(ctx, "extract_iv_rank", lineage, ctx.na_reason or "missing_dependency")})

    rows = grab_list(iv_payload)
    if rows:
        row = rows[0]
        val = safe_float(row.get("iv_rank") or row.get("rank") or row.get("percentile"))
    elif isinstance(iv_payload, dict):
        val = safe_float(iv_payload.get("iv_rank") or iv_payload.get("rank") or iv_payload.get("percentile"))
    else:
        val = None

    if val is None:
        return FeatureBundle({"iv_rank": None}, {"iv": _build_error_meta(ctx, "extract_iv_rank", lineage, "unparseable_rows")})

    if val > 1.0:
        val = val / 100.0
    val = max(min(val, 1.0), 0.0)

    prov = _extract_payload_provenance(iv_payload)
    meta = _build_meta(ctx, "extract_iv_rank", lineage, {**prov, "iv_rank_raw": val, "status": "ok"})
    return FeatureBundle({"iv_rank": val}, {"iv": meta})


def extract_vol_term_slope(ts_payload: Any, ctx: EndpointContext) -> FeatureBundle:
    lineage = _with_output_domain({
        "metric_name": "vol_term_slope",
        "fields_used": ["near_iv", "far_iv", "iv", "tenor"],
        "units_expected": "Implied volatility spread",
        "normalization": "far_iv - near_iv",
        "session_applicability": "RTH",
        "quality_policy": "None if term structure unavailable",
        "criticality": "NON_CRITICAL"
    }, emitted_units="implied_volatility_spread", raw_input_units="Implied Volatility", bounded_output=False, output_domain="unbounded_scalar")

    if is_na(ts_payload) or ctx.freshness_state == "ERROR":
        return FeatureBundle({"vol_term_slope": None}, {"ts": _build_error_meta(ctx, "extract_vol_term_slope", lineage, ctx.na_reason or "missing_dependency")})

    rows = grab_list(ts_payload)
    if len(rows) >= 2:
        scored = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            tenor = safe_float(row.get("tenor") or row.get("days") or row.get("dte"))
            iv = safe_float(row.get("iv") or row.get("volatility") or row.get("implied_volatility"))
            if tenor is None or iv is None:
                continue
            scored.append((tenor, iv))
        scored.sort(key=lambda x: x[0])
        if len(scored) >= 2:
            near_iv = scored[0][1]
            far_iv = scored[-1][1]
        else:
            near_iv = far_iv = None
    elif isinstance(ts_payload, dict):
        near_iv = safe_float(ts_payload.get("near_iv"))
        far_iv = safe_float(ts_payload.get("far_iv"))
    else:
        near_iv = far_iv = None

    if near_iv is None or far_iv is None:
        return FeatureBundle({"vol_term_slope": None}, {"ts": _build_error_meta(ctx, "extract_vol_term_slope", lineage, "unparseable_rows")})

    slope = far_iv - near_iv
    prov = _extract_payload_provenance(ts_payload)
    meta = _build_meta(ctx, "extract_vol_term_slope", lineage, {
        **prov,
        "near_iv": near_iv,
        "far_iv": far_iv,
        "status": "ok",
    })
    return FeatureBundle({"vol_term_slope": slope}, {"ts": meta})


def extract_vol_skew(skew_payload: Any, ctx: EndpointContext) -> FeatureBundle:
    lineage = _with_output_domain({
        "metric_name": "vol_skew",
        "fields_used": ["risk_reversal", "rr", "put_iv", "call_iv"],
        "units_expected": "Skew ratio or risk reversal",
        "normalization": "use explicit risk reversal if present else put_iv - call_iv",
        "session_applicability": "RTH",
        "quality_policy": "None if skew unavailable",
        "criticality": "NON_CRITICAL"
    }, emitted_units="skew_ratio", raw_input_units="Skew Ratio", bounded_output=False, output_domain="unbounded_scalar")

    if is_na(skew_payload) or ctx.freshness_state == "ERROR":
        return FeatureBundle({"vol_skew": None}, {"skew": _build_error_meta(ctx, "extract_vol_skew", lineage, ctx.na_reason or "missing_dependency")})

    rows = grab_list(skew_payload)
    if rows:
        row = rows[0]
    elif isinstance(skew_payload, dict):
        row = skew_payload
    else:
        row = {}

    rr = safe_float(row.get("risk_reversal") or row.get("rr"))
    if rr is None:
        put_iv = safe_float(row.get("put_iv") or row.get("putIV"))
        call_iv = safe_float(row.get("call_iv") or row.get("callIV"))
        if put_iv is not None and call_iv is not None:
            rr = put_iv - call_iv

    if rr is None:
        return FeatureBundle({"vol_skew": None}, {"skew": _build_error_meta(ctx, "extract_vol_skew", lineage, "unparseable_rows")})

    prov = _extract_payload_provenance(skew_payload)
    meta = _build_meta(ctx, "extract_vol_skew", lineage, {**prov, "rr_raw": rr, "status": "ok"})
    return FeatureBundle({"vol_skew": rr}, {"skew": meta})


def extract_darkpool_pressure(dp_payload: Any, ctx: EndpointContext, norm_scale: float = 5_000_000.0) -> FeatureBundle:
    lineage = _with_output_domain({
        "metric_name": "darkpool_pressure",
        "fields_used": ["premium", "notional", "side", "direction", "buy_notional", "sell_notional"],
        "units_expected": "Directional imbalance ratio [-1,1]",
        "normalization": "buy_minus_sell over normalized notional",
        "session_applicability": "RTH",
        "quality_policy": "None if directionality unavailable",
        "criticality": "NON_CRITICAL"
    }, emitted_units="directional_imbalance_ratio", raw_input_units="Notional Flow (USD)", bounded_output=True, expected_bounds=(-1.0, 1.0), output_domain="closed_interval")

    if is_na(dp_payload) or ctx.freshness_state == "ERROR":
        return FeatureBundle({"darkpool_pressure": None}, {"dp": _build_error_meta(ctx, "extract_darkpool_pressure", lineage, ctx.na_reason or "missing_dependency")})

    rows = grab_list(dp_payload)
    buy = sell = 0.0
    valid_count = 0

    if isinstance(dp_payload, dict):
        buy_direct = safe_float(dp_payload.get("buy_notional") or dp_payload.get("buy"))
        sell_direct = safe_float(dp_payload.get("sell_notional") or dp_payload.get("sell"))
        if buy_direct is not None and sell_direct is not None:
            buy = buy_direct
            sell = sell_direct
            valid_count = 1

    if valid_count == 0:
        for r in rows:
            if not isinstance(r, dict):
                continue
            side = str(r.get("side") or r.get("direction") or "").upper().strip()
            val = safe_float(r.get("premium") or r.get("notional") or r.get("value"))
            if val is None:
                continue
            if side in BULLISH_SIDE_ALIASES:
                buy += val
                valid_count += 1
            elif side in BEARISH_SIDE_ALIASES:
                sell += val
                valid_count += 1

    if valid_count == 0:
        return FeatureBundle({"darkpool_pressure": None}, {"dp": _build_error_meta(ctx, "extract_darkpool_pressure", lineage, "unparseable_rows")})

    total = buy + sell
    score = 0.0 if total == 0 else max(min((buy - sell) / total, 1.0), -1.0)
    prov = _extract_payload_provenance(dp_payload)
    meta = _build_meta(ctx, "extract_darkpool_pressure", lineage, {
        **prov,
        "buy_notional": buy,
        "sell_notional": sell,
        "status": "ok",
    })
    return FeatureBundle({"darkpool_pressure": score}, {"dp": meta})


def extract_litflow_pressure(lf_payload: Any, ctx: EndpointContext, norm_scale: float = 5_000_000.0) -> FeatureBundle:
    lineage = _with_output_domain({
        "metric_name": "litflow_pressure",
        "fields_used": ["premium", "notional", "side", "direction", "buy_notional", "sell_notional"],
        "units_expected": "Directional imbalance ratio [-1,1]",
        "normalization": "buy_minus_sell over normalized notional",
        "session_applicability": "RTH",
        "quality_policy": "None if directionality unavailable",
        "criticality": "NON_CRITICAL"
    }, emitted_units="directional_imbalance_ratio", raw_input_units="Notional Flow (USD)", bounded_output=True, expected_bounds=(-1.0, 1.0), output_domain="closed_interval")

    if is_na(lf_payload) or ctx.freshness_state == "ERROR":
        return FeatureBundle({"litflow_pressure": None}, {"lf": _build_error_meta(ctx, "extract_litflow_pressure", lineage, ctx.na_reason or "missing_dependency")})

    rows = grab_list(lf_payload)
    buy = sell = 0.0
    valid_count = 0

    if isinstance(lf_payload, dict):
        buy_direct = safe_float(lf_payload.get("buy_notional") or lf_payload.get("buy"))
        sell_direct = safe_float(lf_payload.get("sell_notional") or lf_payload.get("sell"))
        if buy_direct is not None and sell_direct is not None:
            buy = buy_direct
            sell = sell_direct
            valid_count = 1

    if valid_count == 0:
        for r in rows:
            if not isinstance(r, dict):
                continue
            side = str(r.get("side") or r.get("direction") or "").upper().strip()
            val = safe_float(r.get("premium") or r.get("notional") or r.get("value"))
            if val is None:
                continue
            if side in BULLISH_SIDE_ALIASES:
                buy += val
                valid_count += 1
            elif side in BEARISH_SIDE_ALIASES:
                sell += val
                valid_count += 1

    if valid_count == 0:
        return FeatureBundle({"litflow_pressure": None}, {"lf": _build_error_meta(ctx, "extract_litflow_pressure", lineage, "unparseable_rows")})

    total = buy + sell
    score = 0.0 if total == 0 else max(min((buy - sell) / total, 1.0), -1.0)
    prov = _extract_payload_provenance(lf_payload)
    meta = _build_meta(ctx, "extract_litflow_pressure", lineage, {
        **prov,
        "buy_notional": buy,
        "sell_notional": sell,
        "status": "ok",
    })
    return FeatureBundle({"litflow_pressure": score}, {"lf": meta})


def build_derived_levels(feature_rows: List[FeatureRow], raw_payloads: Dict[str, Any]) -> List[LevelRow]:
    levels: List[LevelRow] = []

    spot = next((fr["feature_value"] for fr in feature_rows if fr["feature_key"] == "spot"), None)
    if spot is None:
        return levels

    if raw_payloads.get("/api/stock/{ticker}/spot-exposures") is not None or raw_payloads.get("/api/stock/{ticker}/spot-exposures/strike") is not None:
        gex_levels = build_gex_levels(
            raw_payloads.get("/api/stock/{ticker}/spot-exposures")
            or raw_payloads.get("/api/stock/{ticker}/spot-exposures/strike"),
            spot=spot,
        )
        for row in gex_levels:
            meta = dict(row.get("meta_json") or {})
            contract = derived_level_usage_contract("gex")
            meta.update(contract)
            levels.append(_level_row(row["level_type"], safe_float(row.get("price")), safe_float(row.get("magnitude")), meta))

    if raw_payloads.get("/api/stock/{ticker}/oi-per-strike") is not None:
        oi_levels = build_oi_walls(raw_payloads.get("/api/stock/{ticker}/oi-per-strike"), spot=spot)
        for row in oi_levels:
            meta = dict(row.get("meta_json") or {})
            contract = derived_level_usage_contract("oi")
            meta.update(contract)
            levels.append(_level_row(row["level_type"], safe_float(row.get("price")), safe_float(row.get("magnitude")), meta))

    if raw_payloads.get("/api/darkpool/{ticker}") is not None:
        dp_levels = build_darkpool_levels(raw_payloads.get("/api/darkpool/{ticker}"), spot=spot)
        for row in dp_levels:
            meta = dict(row.get("meta_json") or {})
            contract = derived_level_usage_contract("darkpool")
            meta.update(contract)
            levels.append(_level_row(row["level_type"], safe_float(row.get("price")), safe_float(row.get("magnitude")), meta))

    return levels


def _merge_meta_for_feature(path: str, ctx: EndpointContext, meta: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(meta)
    _resolve_feature_use_role(merged, path=path)
    _propagate_metric_lineage(merged, ctx)
    return merged


def _bundle_to_feature_rows(
    bundle: FeatureBundle,
    path: str,
    ctx: EndpointContext,
    feature_order: Optional[List[str]] = None,
) -> List[FeatureRow]:
    rows: List[FeatureRow] = []
    ordered_keys = feature_order or list(bundle.features.keys())
    for key in ordered_keys:
        val = bundle.features.get(key)
        # Best-effort meta selection: first meta blob in bundle
        meta_blob = next(iter(bundle.meta.values())) if bundle.meta else {}
        meta = _merge_meta_for_feature(path, ctx, copy.deepcopy(meta_blob))
        rows.append(_feature_row(key, val, meta))
    return rows


def _candidate_from_row(fr: FeatureRow, path: str, payload: Any, ctx: EndpointContext) -> FeatureCandidate:
    freshness_rank, stale_age, path_priority, endpoint_id = _ctx_sort_keys(ctx, payload)
    return FeatureCandidate(
        feature_key=fr["feature_key"],
        feature_value=fr["feature_value"],
        meta_json=fr["meta_json"],
        freshness_rank=freshness_rank,
        stale_age=stale_age,
        path_priority=path_priority,
        endpoint_id=endpoint_id,
        is_none=fr["feature_value"] is None,
    )


def _choose_best_feature_row(candidates: List[FeatureCandidate]) -> Optional[FeatureRow]:
    if not candidates:
        return None

    def sort_key(c: FeatureCandidate) -> Tuple[int, int, int, int]:
        return (c.freshness_rank, c.stale_age, c.path_priority, c.endpoint_id)

    best = min(candidates, key=sort_key)
    return {
        "feature_key": best.feature_key,
        "feature_value": best.feature_value,
        "meta_json": best.meta_json,
    }


def _best_feature_rows(rows: List[FeatureRow], raw_payloads: Dict[str, Any], ctx_map: Dict[str, EndpointContext]) -> List[FeatureRow]:
    by_key: Dict[str, List[FeatureCandidate]] = {}

    for path, payload in raw_payloads.items():
        ctx = ctx_map.get(path)
        if ctx is None:
            continue
        for fr in rows:
            if not isinstance(fr, dict):
                continue
            by_key.setdefault(fr["feature_key"], []).append(_candidate_from_row(fr, path, payload, ctx))

    selected: List[FeatureRow] = []
    for key, cands in by_key.items():
        chosen = _choose_best_feature_row(cands)
        if chosen is not None:
            selected.append(chosen)
    return selected


def _ctx_map_for_payloads(payloads: Dict[str, Tuple[Any, EndpointContext]]) -> Dict[str, EndpointContext]:
    return {path: ctx for path, (_, ctx) in payloads.items()}


def extract_features_from_payloads(payloads: Dict[str, Tuple[Any, EndpointContext]]) -> Tuple[List[FeatureRow], List[LevelRow]]:
    rows: List[FeatureRow] = []
    raw_payloads: Dict[str, Any] = {}

    for path, (payload, ctx) in payloads.items():
        raw_payloads[path] = payload
        if "ohlc" in path:
            f_bundle = extract_price_features(payload, ctx)
            rows.extend(_bundle_to_feature_rows(f_bundle, path, ctx, ["spot"]))
        elif "flow" in path and "strike" not in path:
            f_bundle = extract_smart_whale_pressure(payload, ctx)
            rows.extend(_bundle_to_feature_rows(f_bundle, path, ctx, ["smart_whale_pressure"]))
        elif "greek-exposure" in path and "strike" not in path and "expiry" not in path:
            f_bundle = extract_dealer_greeks(payload, ctx)
            rows.extend(_bundle_to_feature_rows(f_bundle, path, ctx, ["dealer_vanna", "dealer_charm", "net_gamma_exposure_notional"]))
            gex_bundle = extract_gex_sign(payload, ctx)
            rows.extend(_bundle_to_feature_rows(gex_bundle, path, ctx, ["net_gex_sign"]))
        elif "oi-per-strike" in path or "oi-change" in path:
            f_bundle = extract_oi_pressure(payload, ctx)
            rows.extend(_bundle_to_feature_rows(f_bundle, path, ctx, ["oi_pressure"]))
        elif "iv-rank" in path:
            f_bundle = extract_iv_rank(payload, ctx)
            rows.extend(_bundle_to_feature_rows(f_bundle, path, ctx, ["iv_rank"]))
        elif "term-structure" in path:
            f_bundle = extract_vol_term_slope(payload, ctx)
            rows.extend(_bundle_to_feature_rows(f_bundle, path, ctx, ["vol_term_slope"]))
        elif "risk-reversal-skew" in path:
            f_bundle = extract_vol_skew(payload, ctx)
            rows.extend(_bundle_to_feature_rows(f_bundle, path, ctx, ["vol_skew"]))
        elif "/api/darkpool/" in path:
            f_bundle = extract_darkpool_pressure(payload, ctx)
            rows.extend(_bundle_to_feature_rows(f_bundle, path, ctx, ["darkpool_pressure"]))
        elif "/api/lit-flow/" in path:
            f_bundle = extract_litflow_pressure(payload, ctx)
            rows.extend(_bundle_to_feature_rows(f_bundle, path, ctx, ["litflow_pressure"]))

    selected_rows = _best_feature_rows(rows, raw_payloads, _ctx_map_for_payloads(payloads))
    levels = build_derived_levels(selected_rows, raw_payloads)
    return selected_rows, levels


def summarize_feature_quality(feature_rows: List[FeatureRow]) -> Dict[str, Any]:
    missing = 0
    degraded = 0
    decision_relevant_missing = 0
    decision_relevant_stale = 0

    for fr in feature_rows:
        meta = fr.get("meta_json") or {}
        feature_value = fr.get("feature_value")
        role = str(meta.get("feature_use_role") or "signal-critical")
        decision_path = bool(meta.get("decision_path", role == "signal-critical"))
        missing_affects_confidence = bool(meta.get("missing_affects_confidence", decision_path))
        stale_affects_confidence = bool(meta.get("stale_affects_confidence", decision_path))
        freshness_state = str(meta.get("freshness_state") or "")

        if feature_value is None:
            missing += 1
            if decision_path and missing_affects_confidence:
                decision_relevant_missing += 1

        confidence_impact = str(meta.get("confidence_impact") or "").upper()
        if confidence_impact == "DEGRADED":
            degraded += 1
            if freshness_state == "STALE_CARRY" and decision_path and stale_affects_confidence:
                decision_relevant_stale += 1
        elif freshness_state == "STALE_CARRY":
            degraded += 1
            if decision_path and stale_affects_confidence:
                decision_relevant_stale += 1

    return {
        "missing_feature_count": missing,
        "degraded_feature_count": degraded,
        "decision_relevant_missing_feature_count": decision_relevant_missing,
        "decision_relevant_stale_feature_count": decision_relevant_stale,
    }


def feature_keys(feature_rows: List[FeatureRow]) -> List[str]:
    return [fr["feature_key"] for fr in feature_rows]


def feature_map(feature_rows: List[FeatureRow]) -> Dict[str, Optional[float]]:
    return {fr["feature_key"]: fr["feature_value"] for fr in feature_rows}


def meta_map(feature_rows: List[FeatureRow]) -> Dict[str, Dict[str, Any]]:
    return {fr["feature_key"]: fr["meta_json"] for fr in feature_rows}


def serialize_feature_rows(feature_rows: List[FeatureRow]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for fr in feature_rows:
        meta = copy.deepcopy(fr["meta_json"])
        rows.append({
            "feature_key": fr["feature_key"],
            "feature_value": fr["feature_value"],
            "meta_json": meta,
        })
    return rows


def serialize_level_rows(level_rows: List[LevelRow]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for lr in level_rows:
        meta = copy.deepcopy(lr["meta_json"])
        rows.append({
            "level_type": lr["level_type"],
            "price": lr["price"],
            "magnitude": lr["magnitude"],
            "meta_json": meta,
        })
    return rows


def deserialize_feature_rows(payload: List[Dict[str, Any]]) -> List[FeatureRow]:
    return [
        {
            "feature_key": str(item["feature_key"]),
            "feature_value": safe_float(item.get("feature_value")) if item.get("feature_value") is not None else None,
            "meta_json": dict(item.get("meta_json") or {}),
        }
        for item in payload
    ]


def deserialize_level_rows(payload: List[Dict[str, Any]]) -> List[LevelRow]:
    return [
        {
            "level_type": str(item["level_type"]),
            "price": safe_float(item.get("price")) if item.get("price") is not None else None,
            "magnitude": safe_float(item.get("magnitude")) if item.get("magnitude") is not None else None,
            "meta_json": dict(item.get("meta_json") or {}),
        }
        for item in payload
    ]


def feature_summary(feature_rows: List[FeatureRow]) -> Dict[str, Any]:
    fmap = feature_map(feature_rows)
    q = summarize_feature_quality(feature_rows)
    return {
        "n_features": len(feature_rows),
        "feature_keys": list(fmap.keys()),
        "missing_feature_count": q["missing_feature_count"],
        "degraded_feature_count": q["degraded_feature_count"],
    }


def level_summary(level_rows: List[LevelRow]) -> Dict[str, Any]:
    return {
        "n_levels": len(level_rows),
        "level_types": [lr["level_type"] for lr in level_rows],
    }


def extract_feature_bundles(payloads: Dict[str, Tuple[Any, EndpointContext]]) -> Dict[str, FeatureBundle]:
    bundles: Dict[str, FeatureBundle] = {}
    for path, (payload, ctx) in payloads.items():
        if "ohlc" in path:
            bundles[path] = extract_price_features(payload, ctx)
        elif "flow" in path and "strike" not in path:
            bundles[path] = extract_smart_whale_pressure(payload, ctx)
        elif "greek-exposure" in path and "strike" not in path and "expiry" not in path:
            bundles[path] = extract_dealer_greeks(payload, ctx)
        elif "oi-per-strike" in path or "oi-change" in path:
            bundles[path] = extract_oi_pressure(payload, ctx)
        elif "iv-rank" in path:
            bundles[path] = extract_iv_rank(payload, ctx)
        elif "term-structure" in path:
            bundles[path] = extract_vol_term_slope(payload, ctx)
        elif "risk-reversal-skew" in path:
            bundles[path] = extract_vol_skew(payload, ctx)
        elif "/api/darkpool/" in path:
            bundles[path] = extract_darkpool_pressure(payload, ctx)
        elif "/api/lit-flow/" in path:
            bundles[path] = extract_litflow_pressure(payload, ctx)
    return bundles


def merge_feature_bundles(bundles: Dict[str, FeatureBundle], ctx_map: Dict[str, EndpointContext]) -> Tuple[List[FeatureRow], List[LevelRow]]:
    feature_rows: List[FeatureRow] = []
    raw_payloads: Dict[str, Any] = {}

    for path, bundle in bundles.items():
        ctx = ctx_map[path]
        raw_payloads[path] = None
        if "ohlc" in path:
            feature_rows.extend(_bundle_to_feature_rows(bundle, path, ctx, ["spot"]))
        elif "flow" in path and "strike" not in path:
            feature_rows.extend(_bundle_to_feature_rows(bundle, path, ctx, ["smart_whale_pressure"]))
        elif "greek-exposure" in path and "strike" not in path and "expiry" not in path:
            feature_rows.extend(_bundle_to_feature_rows(bundle, path, ctx, ["dealer_vanna", "dealer_charm", "net_gamma_exposure_notional"]))
        elif "oi-per-strike" in path or "oi-change" in path:
            feature_rows.extend(_bundle_to_feature_rows(bundle, path, ctx, ["oi_pressure"]))
        elif "iv-rank" in path:
            feature_rows.extend(_bundle_to_feature_rows(bundle, path, ctx, ["iv_rank"]))
        elif "term-structure" in path:
            feature_rows.extend(_bundle_to_feature_rows(bundle, path, ctx, ["vol_term_slope"]))
        elif "risk-reversal-skew" in path:
            feature_rows.extend(_bundle_to_feature_rows(bundle, path, ctx, ["vol_skew"]))
        elif "/api/darkpool/" in path:
            feature_rows.extend(_bundle_to_feature_rows(bundle, path, ctx, ["darkpool_pressure"]))
        elif "/api/lit-flow/" in path:
            feature_rows.extend(_bundle_to_feature_rows(bundle, path, ctx, ["litflow_pressure"]))

    selected_rows = _best_feature_rows(feature_rows, raw_payloads, ctx_map)
    levels = build_derived_levels(selected_rows, raw_payloads)
    return selected_rows, levels


def _rows_to_feature_candidates(rows: List[FeatureRow], payloads: Dict[str, Any], ctx_map: Dict[str, EndpointContext]) -> List[FeatureCandidate]:
    candidates: List[FeatureCandidate] = []
    for path, payload in payloads.items():
        ctx = ctx_map[path]
        for fr in rows:
            candidates.append(_candidate_from_row(fr, path, payload, ctx))
    return candidates


def _score_feature_candidate(fc: FeatureCandidate) -> Tuple[int, int, int, int]:
    return (fc.freshness_rank, fc.stale_age, fc.path_priority, fc.endpoint_id)


def select_best_features(rows: List[FeatureRow], payloads: Dict[str, Any], ctx_map: Dict[str, EndpointContext]) -> List[FeatureRow]:
    by_key: Dict[str, List[FeatureCandidate]] = {}
    for path, payload in payloads.items():
        ctx = ctx_map[path]
        for fr in rows:
            by_key.setdefault(fr["feature_key"], []).append(_candidate_from_row(fr, path, payload, ctx))

    out: List[FeatureRow] = []
    for _, cands in by_key.items():
        cands.sort(key=_score_feature_candidate)
        if cands:
            best = cands[0]
            out.append({
                "feature_key": best.feature_key,
                "feature_value": best.feature_value,
                "meta_json": best.meta_json,
            })
    return out


def feature_value_or_none(feature_rows: List[FeatureRow], key: str) -> Optional[float]:
    for fr in feature_rows:
        if fr["feature_key"] == key:
            return fr["feature_value"]
    return None


def feature_meta_or_empty(feature_rows: List[FeatureRow], key: str) -> Dict[str, Any]:
    for fr in feature_rows:
        if fr["feature_key"] == key:
            return fr["meta_json"]
    return {}


def _is_stale_meta(meta: Dict[str, Any]) -> bool:
    freshness_state = str(meta.get("freshness_state") or "")
    return freshness_state == "STALE_CARRY"


def _is_missing_feature_row(fr: FeatureRow) -> bool:
    return fr.get("feature_value") is None


def _feature_decision_relevant(fr: FeatureRow) -> bool:
    meta = fr.get("meta_json") or {}
    return bool(meta.get("decision_path", True))


def decision_relevant_missing(feature_rows: List[FeatureRow]) -> List[str]:
    keys: List[str] = []
    for fr in feature_rows:
        meta = fr.get("meta_json") or {}
        if fr.get("feature_value") is None and bool(meta.get("decision_path", True)) and bool(meta.get("missing_affects_confidence", True)):
            keys.append(fr["feature_key"])
    return keys


def decision_relevant_stale(feature_rows: List[FeatureRow]) -> List[str]:
    keys: List[str] = []
    for fr in feature_rows:
        meta = fr.get("meta_json") or {}
        freshness_state = str(meta.get("freshness_state") or "")
        confidence_impact = str(meta.get("confidence_impact") or "").upper()
        if freshness_state == "STALE_CARRY" and bool(meta.get("decision_path", True)) and bool(meta.get("stale_affects_confidence", True)):
            keys.append(fr["feature_key"])
        elif confidence_impact == "DEGRADED" and freshness_state == "STALE_CARRY" and bool(meta.get("decision_path", True)) and bool(meta.get("stale_affects_confidence", True)):
            keys.append(fr["feature_key"])
    return keys


def _copy_rows(rows: List[FeatureRow]) -> List[FeatureRow]:
    return [
        {
            "feature_key": fr["feature_key"],
            "feature_value": fr["feature_value"],
            "meta_json": copy.deepcopy(fr["meta_json"]),
        }
        for fr in rows
    ]


def _copy_levels(levels: List[LevelRow]) -> List[LevelRow]:
    return [
        {
            "level_type": lr["level_type"],
            "price": lr["price"],
            "magnitude": lr["magnitude"],
            "meta_json": copy.deepcopy(lr["meta_json"]),
        }
        for lr in levels
    ]


def filter_feature_rows(feature_rows: List[FeatureRow], keys: Iterable[str]) -> List[FeatureRow]:
    keyset = set(keys)
    return [fr for fr in feature_rows if fr["feature_key"] in keyset]


def filter_level_rows(level_rows: List[LevelRow], level_types: Iterable[str]) -> List[LevelRow]:
    lset = set(level_types)
    return [lr for lr in level_rows if lr["level_type"] in lset]


def sort_feature_rows(feature_rows: List[FeatureRow]) -> List[FeatureRow]:
    return sorted(feature_rows, key=lambda r: r["feature_key"])


def sort_level_rows(level_rows: List[LevelRow]) -> List[LevelRow]:
    return sorted(level_rows, key=lambda r: (r["level_type"], r["price"] if r["price"] is not None else 0.0))


def has_feature(feature_rows: List[FeatureRow], key: str) -> bool:
    return any(fr["feature_key"] == key for fr in feature_rows)


def has_level(level_rows: List[LevelRow], level_type: str) -> bool:
    return any(lr["level_type"] == level_type for lr in level_rows)


def missing_feature_keys(feature_rows: List[FeatureRow], required_keys: Iterable[str]) -> List[str]:
    present = set(feature_keys(feature_rows))
    return [k for k in required_keys if k not in present]


def count_missing_values(feature_rows: List[FeatureRow]) -> int:
    return sum(1 for fr in feature_rows if fr["feature_value"] is None)


def count_degraded(feature_rows: List[FeatureRow]) -> int:
    count = 0
    for fr in feature_rows:
        meta = fr.get("meta_json") or {}
        if str(meta.get("confidence_impact") or "").upper() == "DEGRADED":
            count += 1
        elif str(meta.get("freshness_state") or "") == "STALE_CARRY":
            count += 1
    return count


def flatten_feature_meta(feature_rows: List[FeatureRow]) -> Dict[str, Dict[str, Any]]:
    return {fr["feature_key"]: copy.deepcopy(fr["meta_json"]) for fr in feature_rows}


def flatten_level_meta(level_rows: List[LevelRow]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for lr in level_rows:
        out.setdefault(lr["level_type"], []).append(copy.deepcopy(lr["meta_json"]))
    return out


def build_feature_index(feature_rows: List[FeatureRow]) -> Dict[str, FeatureRow]:
    return {fr["feature_key"]: fr for fr in feature_rows}


def build_level_index(level_rows: List[LevelRow]) -> Dict[str, List[LevelRow]]:
    idx: Dict[str, List[LevelRow]] = {}
    for lr in level_rows:
        idx.setdefault(lr["level_type"], []).append(lr)
    return idx


def validate_feature_rows(feature_rows: List[FeatureRow]) -> List[str]:
    errors: List[str] = []
    for fr in feature_rows:
        if "feature_key" not in fr:
            errors.append("missing_feature_key")
            continue
        if "meta_json" not in fr:
            errors.append(f"{fr.get('feature_key', 'unknown')}:missing_meta_json")
            continue
        meta = fr["meta_json"]
        if not isinstance(meta, dict):
            errors.append(f"{fr['feature_key']}:meta_not_dict")
            continue
        if "feature_use_role" not in meta:
            errors.append(f"{fr['feature_key']}:missing_feature_use_role")
        if "feature_use_contract_version" not in meta:
            errors.append(f"{fr['feature_key']}:missing_feature_use_contract_version")
    return errors


def validate_level_rows(level_rows: List[LevelRow]) -> List[str]:
    errors: List[str] = []
    for lr in level_rows:
        if "level_type" not in lr:
            errors.append("missing_level_type")
        if "meta_json" not in lr or not isinstance(lr["meta_json"], dict):
            errors.append(f"{lr.get('level_type', 'unknown')}:invalid_meta_json")
    return errors


def _row_from_map(key: str, fmap: Dict[str, Optional[float]], mmap: Dict[str, Dict[str, Any]]) -> FeatureRow:
    return {
        "feature_key": key,
        "feature_value": fmap.get(key),
        "meta_json": copy.deepcopy(mmap.get(key, {})),
    }


def combine_feature_maps(fmaps: List[Dict[str, Optional[float]]], mmaps: List[Dict[str, Dict[str, Any]]]) -> List[FeatureRow]:
    keys: List[str] = []
    for fmap in fmaps:
        for k in fmap.keys():
            if k not in keys:
                keys.append(k)

    rows: List[FeatureRow] = []
    for k in keys:
        value = None
        meta: Dict[str, Any] = {}
        for fmap, mmap in zip(fmaps, mmaps):
            if k in fmap and value is None:
                value = fmap.get(k)
                meta = copy.deepcopy(mmap.get(k, {}))
        rows.append({"feature_key": k, "feature_value": value, "meta_json": meta})
    return rows


def _level_from_map(level_type: str, price: Optional[float], magnitude: Optional[float], meta: Dict[str, Any]) -> LevelRow:
    return {
        "level_type": level_type,
        "price": price,
        "magnitude": magnitude,
        "meta_json": copy.deepcopy(meta),
    }


def is_valid_feature_value(value: Optional[float]) -> bool:
    if value is None:
        return False
    return math.isfinite(float(value))


def is_valid_price(value: Optional[float]) -> bool:
    if value is None:
        return False
    v = float(value)
    return math.isfinite(v) and v > 0


def coerce_feature_value(value: Any) -> Optional[float]:
    return safe_float(value)


def feature_keys_sorted(feature_rows: List[FeatureRow]) -> List[str]:
    return sorted(feature_keys(feature_rows))


def level_types_sorted(level_rows: List[LevelRow]) -> List[str]:
    return sorted({lr["level_type"] for lr in level_rows})


def select_feature_rows(feature_rows: List[FeatureRow], predicate) -> List[FeatureRow]:
    return [fr for fr in feature_rows if predicate(fr)]


def select_level_rows(level_rows: List[LevelRow], predicate) -> List[LevelRow]:
    return [lr for lr in level_rows if predicate(lr)]


def map_feature_rows(feature_rows: List[FeatureRow], mapper) -> List[FeatureRow]:
    return [mapper(fr) for fr in feature_rows]


def map_level_rows(level_rows: List[LevelRow], mapper) -> List[LevelRow]:
    return [mapper(lr) for lr in level_rows]


def any_missing_decision_relevant(feature_rows: List[FeatureRow]) -> bool:
    return len(decision_relevant_missing(feature_rows)) > 0


def any_stale_decision_relevant(feature_rows: List[FeatureRow]) -> bool:
    return len(decision_relevant_stale(feature_rows)) > 0


def count_decision_relevant(feature_rows: List[FeatureRow]) -> int:
    return sum(1 for fr in feature_rows if _feature_decision_relevant(fr))


def count_decision_relevant_missing(feature_rows: List[FeatureRow]) -> int:
    return len(decision_relevant_missing(feature_rows))


def count_decision_relevant_stale(feature_rows: List[FeatureRow]) -> int:
    return len(decision_relevant_stale(feature_rows))


def feature_row_by_key(feature_rows: List[FeatureRow], key: str) -> Optional[FeatureRow]:
    for fr in feature_rows:
        if fr["feature_key"] == key:
            return fr
    return None


def level_rows_by_type(level_rows: List[LevelRow], level_type: str) -> List[LevelRow]:
    return [lr for lr in level_rows if lr["level_type"] == level_type]


def ensure_feature_use_contract(feature_rows: List[FeatureRow]) -> List[FeatureRow]:
    rows = _copy_rows(feature_rows)
    for fr in rows:
        _resolve_feature_use_role(fr["meta_json"], path=fr["meta_json"].get("source_endpoints", [{}])[0].get("path") if isinstance(fr["meta_json"].get("source_endpoints"), list) and fr["meta_json"].get("source_endpoints") else None)
    return rows


def ensure_metric_lineage(feature_rows: List[FeatureRow], ctx: Optional[EndpointContext] = None) -> List[FeatureRow]:
    rows = _copy_rows(feature_rows)
    for fr in rows:
        meta = fr["meta_json"]
        lineage = meta.get("metric_lineage")
        if not isinstance(lineage, dict):
            meta["metric_lineage"] = {}
        if ctx is not None:
            _propagate_metric_lineage(meta, ctx)
    return rows


def add_feature_use_role(feature_rows: List[FeatureRow], role: str) -> List[FeatureRow]:
    rows = _copy_rows(feature_rows)
    for fr in rows:
        _resolve_feature_use_role(fr["meta_json"], role_override=role)
    return rows


def feature_row_to_dict(fr: FeatureRow) -> Dict[str, Any]:
    return {
        "feature_key": fr["feature_key"],
        "feature_value": fr["feature_value"],
        "meta_json": copy.deepcopy(fr["meta_json"]),
    }


def level_row_to_dict(lr: LevelRow) -> Dict[str, Any]:
    return {
        "level_type": lr["level_type"],
        "price": lr["price"],
        "magnitude": lr["magnitude"],
        "meta_json": copy.deepcopy(lr["meta_json"]),
    }


def feature_rows_to_dicts(feature_rows: List[FeatureRow]) -> List[Dict[str, Any]]:
    return [feature_row_to_dict(fr) for fr in feature_rows]


def level_rows_to_dicts(level_rows: List[LevelRow]) -> List[Dict[str, Any]]:
    return [level_row_to_dict(lr) for lr in level_rows]


def dicts_to_feature_rows(items: List[Dict[str, Any]]) -> List[FeatureRow]:
    return deserialize_feature_rows(items)


def dicts_to_level_rows(items: List[Dict[str, Any]]) -> List[LevelRow]:
    return deserialize_level_rows(items)


def update_feature_meta(feature_rows: List[FeatureRow], key: str, updates: Dict[str, Any]) -> List[FeatureRow]:
    rows = _copy_rows(feature_rows)
    for fr in rows:
        if fr["feature_key"] == key:
            fr["meta_json"].update(copy.deepcopy(updates))
            break
    return rows


def update_level_meta(level_rows: List[LevelRow], level_type: str, updates: Dict[str, Any]) -> List[LevelRow]:
    rows = _copy_levels(level_rows)
    for lr in rows:
        if lr["level_type"] == level_type:
            lr["meta_json"].update(copy.deepcopy(updates))
    return rows


def feature_values_vector(feature_rows: List[FeatureRow], ordered_keys: List[str]) -> List[Optional[float]]:
    fmap = feature_map(feature_rows)
    return [fmap.get(k) for k in ordered_keys]


def assert_feature_keys(feature_rows: List[FeatureRow], expected_keys: Iterable[str]) -> None:
    present = set(feature_keys(feature_rows))
    expected = set(expected_keys)
    missing = expected - present
    if missing:
        raise AssertionError(f"missing_feature_keys={sorted(missing)}")


def assert_level_types(level_rows: List[LevelRow], expected_types: Iterable[str]) -> None:
    present = {lr["level_type"] for lr in level_rows}
    expected = set(expected_types)
    missing = expected - present
    if missing:
        raise AssertionError(f"missing_level_types={sorted(missing)}")


def extract_features(
    ohlc_payload: Any = None,
    flow_payload: Any = None,
    greek_payload: Any = None,
    oi_payload: Any = None,
    iv_payload: Any = None,
    ts_payload: Any = None,
    skew_payload: Any = None,
    darkpool_payload: Any = None,
    litflow_payload: Any = None,
    *,
    ohlc_ctx: Optional[EndpointContext] = None,
    flow_ctx: Optional[EndpointContext] = None,
    greek_ctx: Optional[EndpointContext] = None,
    oi_ctx: Optional[EndpointContext] = None,
    iv_ctx: Optional[EndpointContext] = None,
    ts_ctx: Optional[EndpointContext] = None,
    skew_ctx: Optional[EndpointContext] = None,
    darkpool_ctx: Optional[EndpointContext] = None,
    litflow_ctx: Optional[EndpointContext] = None,
) -> Tuple[List[FeatureRow], List[LevelRow]]:
    payloads: Dict[str, Tuple[Any, EndpointContext]] = {}

    def _default_ctx(path: str) -> EndpointContext:
        return EndpointContext(path=path, method="GET", freshness_state="FRESH", stale_age_min=0, na_reason=None)

    if ohlc_payload is not None:
        payloads["/api/stock/{ticker}/ohlc/{candle_size}"] = (ohlc_payload, ohlc_ctx or _default_ctx("/api/stock/{ticker}/ohlc/{candle_size}"))
    if flow_payload is not None:
        payloads["/api/stock/{ticker}/flow-recent"] = (flow_payload, flow_ctx or _default_ctx("/api/stock/{ticker}/flow-recent"))
    if greek_payload is not None:
        payloads["/api/stock/{ticker}/greek-exposure"] = (greek_payload, greek_ctx or _default_ctx("/api/stock/{ticker}/greek-exposure"))
    if oi_payload is not None:
        payloads["/api/stock/{ticker}/oi-per-strike"] = (oi_payload, oi_ctx or _default_ctx("/api/stock/{ticker}/oi-per-strike"))
    if iv_payload is not None:
        payloads["/api/stock/{ticker}/iv-rank"] = (iv_payload, iv_ctx or _default_ctx("/api/stock/{ticker}/iv-rank"))
    if ts_payload is not None:
        payloads["/api/stock/{ticker}/volatility/term-structure"] = (ts_payload, ts_ctx or _default_ctx("/api/stock/{ticker}/volatility/term-structure"))
    if skew_payload is not None:
        payloads["/api/stock/{ticker}/historical-risk-reversal-skew"] = (skew_payload, skew_ctx or _default_ctx("/api/stock/{ticker}/historical-risk-reversal-skew"))
    if darkpool_payload is not None:
        payloads["/api/darkpool/{ticker}"] = (darkpool_payload, darkpool_ctx or _default_ctx("/api/darkpool/{ticker}"))
    if litflow_payload is not None:
        payloads["/api/lit-flow/{ticker}"] = (litflow_payload, litflow_ctx or _default_ctx("/api/lit-flow/{ticker}"))

    return extract_features_from_payloads(payloads)
