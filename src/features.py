from __future__ import annotations
import copy
import datetime
import math
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Mapping
from dataclasses import dataclass

from .na import safe_float, is_na, grab_list
from .endpoint_truth import EndpointContext
from .analytics import build_gex_levels

_grab_list = grab_list 
_as_float = safe_float 


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
    "/api/stock/{ticker}/ohlc/{candle_size}": 1
}


def _normalize_signed(x: Optional[float], *, scale: float) -> Optional[float]:
    if scale == 0: 
        return None 
    val = safe_float(x)
    if val is None or not math.isfinite(val): 
        return None
    return max(-1.0, min(1.0, val / scale))


def _parse_strict_ts(row: dict, key: str) -> float:
    """Helper to deterministically extract explicit timestamps."""
    ts_val = row.get(key)
    if isinstance(ts_val, (int, float)): 
        return float(ts_val)
    if isinstance(ts_val, str):
        try:
            return datetime.datetime.fromisoformat(ts_val.replace('Z', '+00:00')).timestamp()
        except ValueError:
            pass
    return 0.0


def _build_meta(
    ctx: EndpointContext, 
    extractor_name: str, 
    lineage: Dict[str, Any], 
    details: Dict[str, Any] = None
) -> Dict[str, Any]:
    
    d = {"extractor": extractor_name}
    if details: 
        d.update(details)
        
    full_lineage = {
        "metric_name": lineage.get("metric_name", "unknown"),
        "source_path": ctx.path,
        "fields_used": lineage.get("fields_used", []),
        "units_expected": lineage.get("units_expected", "unknown"),
        "normalization": lineage.get("normalization", "none"),
        "session_applicability": lineage.get("session_applicability", "PRE/RTH/AFT"),
        "quality_policy": lineage.get("quality_policy", "None on missing"),
        "endpoint_asof_ts_utc": ctx.endpoint_asof_ts_utc.isoformat() if getattr(ctx, "endpoint_asof_ts_utc", None) else None,
        "alignment_delta_sec": getattr(ctx, "alignment_delta_sec", None)
    }
    
    return {
        "source_endpoints": [{
            "method": ctx.method,
            "path": ctx.path,
            "operation_id": ctx.operation_id,
            "endpoint_id": ctx.endpoint_id,
            "used_event_id": ctx.used_event_id,
            "signature": ctx.signature
        }],
        "freshness_state": ctx.freshness_state,
        "stale_age_min": ctx.stale_age_min,
        "na_reason": ctx.na_reason,
        "metric_lineage": full_lineage,
        "details": d
    }


def _build_error_meta(
    ctx: EndpointContext, 
    extractor_name: str, 
    lineage: Dict[str, Any], 
    na_reason: str
) -> Dict[str, Any]:
    meta = _build_meta(ctx, extractor_name, lineage)
    meta["freshness_state"] = "ERROR"
    meta["na_reason"] = na_reason
    return meta


# --- Extractors ---

def extract_price_features(ohlc_payload: Any, ctx: EndpointContext) -> FeatureBundle:
    lineage = {
        "metric_name": "spot",
        "fields_used": ["close", "t"],
        "units_expected": "USD",
        "normalization": "none",
        "session_applicability": "PRE/RTH/AFT",
        "quality_policy": "None if missing required explicit keys"
    }
    
    if is_na(ohlc_payload) or ctx.freshness_state == "ERROR":
        return FeatureBundle({"spot": None}, {"price": _build_error_meta(ctx, "extract_price_features", lineage, ctx.na_reason or "missing_dependency")})
        
    rows = grab_list(ohlc_payload)
    if not rows:
        return FeatureBundle({"spot": None}, {"price": _build_error_meta(ctx, "extract_price_features", lineage, "no_rows")})
    
    latest_row = max(rows, key=lambda r: _parse_strict_ts(r, "t"))
    
    if "close" not in latest_row:
        return FeatureBundle({"spot": None}, {"price": _build_error_meta(ctx, "extract_price_features", lineage, "missing_explicit_close_field")})
        
    close_val = safe_float(latest_row["close"])
    return FeatureBundle({"spot": close_val}, {"price": _build_meta(ctx, "extract_price_features", lineage, {"last_ts": latest_row.get("t")})})


def extract_smart_whale_pressure(flow_payload: Any, ctx: EndpointContext, min_premium: float = 10000.0, max_dte: float = 14.0, norm_scale: float = 500_000.0) -> FeatureBundle:
    lineage = {
        "metric_name": "smart_whale_pressure",
        "fields_used": ["premium", "dte", "side", "put_call"],
        "units_expected": "Net Premium Flow (USD)",
        "normalization": f"normalize_signed [-1, 1] by {norm_scale}",
        "session_applicability": "RTH",
        "quality_policy": "None on filtered zeros to avoid false baseline certainty"
    }
    
    if is_na(flow_payload) or ctx.freshness_state == "ERROR":
        return FeatureBundle({"smart_whale_pressure": None}, {"flow": _build_error_meta(ctx, "extract_smart_whale_pressure", lineage, ctx.na_reason or "missing_dependency")})

    trades = grab_list(flow_payload)
    if not trades and isinstance(flow_payload, dict) and "data" in flow_payload:
        trades = flow_payload["data"]
        
    if trades and not all(isinstance(t, dict) for t in trades):
        return FeatureBundle({"smart_whale_pressure": None}, {"flow": _build_error_meta(ctx, "extract_smart_whale_pressure", lineage, "schema_non_dict_rows")})

    if not trades:
        meta = _build_meta(ctx, "extract_smart_whale_pressure", lineage, {"status": "computed_zero_from_empty_valid", "n_trades": 0})
        return FeatureBundle({"smart_whale_pressure": None}, {"flow": meta})

    whale_call, whale_put = 0.0, 0.0
    valid_count, skip_missing_fields, skip_bad_type, skip_bad_side, skip_threshold = 0, 0, 0, 0, 0
    
    pc_map = {"C": "CALL", "CALL": "CALL", "CALLS": "CALL", "P": "PUT", "PUT": "PUT", "PUTS": "PUT"}
    side_map = {"ASK": "BULL", "BUY": "BULL", "BULLISH": "BULL", "BOT": "BULL", "BID": "BEAR", "SELL": "BEAR", "BEARISH": "BEAR", "SOLD": "BEAR"}
    
    for t in trades:
        prem = safe_float(t.get("premium"))
        dte = safe_float(t.get("dte"))
        side_raw = t.get("side")
        pc_raw = t.get("put_call")
        
        if prem is None or dte is None or is_na(side_raw) or is_na(pc_raw):
            skip_missing_fields += 1
            continue
            
        pc_norm = pc_map.get(str(pc_raw).upper().strip())
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

    if valid_count == 0:
        meta = _build_meta(ctx, "extract_smart_whale_pressure", lineage, {
            "status": "filtered_zero_treated_as_unknown", 
            "n_raw_trades": len(trades), 
            "skipped_threshold": skip_threshold, 
            "confidence_impact": "DEGRADED"
        })
        meta["freshness_state"] = "EMPTY_VALID"
        meta["na_reason"] = "no_trades_met_policy_thresholds"
        return FeatureBundle({"smart_whale_pressure": None}, {"flow": meta})

    net = whale_call - whale_put
    meta = _build_meta(ctx, "extract_smart_whale_pressure", lineage, {"net_prem": net, "n_valid": valid_count, "n_raw": len(trades)})
    return FeatureBundle({"smart_whale_pressure": _normalize_signed(net, scale=norm_scale)}, {"flow": meta})


def extract_dealer_greeks(greek_payload: Any, ctx: EndpointContext, norm_scale: float = 1_000_000_000.0) -> FeatureBundle:
    keys = ["dealer_vanna", "dealer_charm", "net_gamma_exposure_notional"]
    lineage = {
        "metric_name": "dealer_greeks",
        "fields_used": ["vanna_exposure", "charm_exposure", "gamma_exposure", "date"],
        "units_expected": "Notional Exposure (USD)",
        "normalization": f"normalize_signed [-1, 1] by {norm_scale}",
        "session_applicability": "PRE/RTH",
        "quality_policy": "None on missing"
    }
    
    if is_na(greek_payload) or ctx.freshness_state == "ERROR":
        return FeatureBundle({k: None for k in keys}, {"greeks": _build_error_meta(ctx, "extract_dealer_greeks", lineage, ctx.na_reason or "missing_dependency")})

    rows = grab_list(greek_payload)
    if not rows:
        return FeatureBundle({k: None for k in keys}, {"greeks": _build_error_meta(ctx, "extract_dealer_greeks", lineage, "no_rows")})

    latest = max(rows, key=lambda r: _parse_strict_ts(r, "date"))
    
    meta = _build_meta(ctx, "extract_dealer_greeks", lineage, {"ts": latest.get("date"), "scale_used": norm_scale})
    
    return FeatureBundle({
        "dealer_vanna": _normalize_signed(safe_float(latest.get("vanna_exposure")), scale=norm_scale),
        "dealer_charm": _normalize_signed(safe_float(latest.get("charm_exposure")), scale=norm_scale),
        "net_gamma_exposure_notional": _normalize_signed(safe_float(latest.get("gamma_exposure")), scale=norm_scale)
    }, {"greeks": meta})


def extract_gex_sign(spot_exposures_payload: Any, ctx: EndpointContext) -> FeatureBundle:
    lineage = {
        "metric_name": "net_gex_sign",
        "fields_used": ["gamma_exposure"],
        "units_expected": "Sign (+1.0, 0.0, -1.0)",
        "normalization": "Directional sign clamping",
        "session_applicability": "PRE/RTH",
        "quality_policy": "None on missing exposure fields"
    }
    
    if is_na(spot_exposures_payload) or ctx.freshness_state == "ERROR":
        return FeatureBundle({"net_gex_sign": None}, {"gex": _build_error_meta(ctx, "extract_gex_sign", lineage, ctx.na_reason or "missing_dependency")})

    rows = grab_list(spot_exposures_payload)
    if not rows:
        return FeatureBundle({"net_gex_sign": None}, {"gex": _build_error_meta(ctx, "extract_gex_sign", lineage, "no_rows")})
    
    tot_gamma = 0.0
    valid_rows = 0
    for r in rows:
        g = safe_float(r.get("gamma_exposure"))
        if g is not None:
            tot_gamma += g
            valid_rows += 1
            
    if valid_rows == 0:
        return FeatureBundle({"net_gex_sign": None}, {"gex": _build_error_meta(ctx, "extract_gex_sign", lineage, "missing_gamma_exposure_fields")})
    
    if abs(tot_gamma) <= 1e-9: 
        sign = 0.0
    else: 
        sign = 1.0 if tot_gamma > 0 else -1.0
        
    meta = _build_meta(ctx, "extract_gex_sign", lineage, {"total": tot_gamma, "n_strikes": valid_rows})
    return FeatureBundle({"net_gex_sign": sign}, {"gex": meta})


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
    "/api/stock/{ticker}/oi-per-strike",
    "/api/stock/{ticker}/oi-change",
    "/api/stock/{ticker}/volatility/term-structure",
    "/api/stock/{ticker}/interpolated-iv",
    "/api/stock/{ticker}/volatility/realized",
    "/api/stock/{ticker}/iv-rank",
    "/api/darkpool/{ticker}",
    "/api/lit-flow/{ticker}",
    "/api/stock/{ticker}/option/stock-price-levels",
    "/api/stock/{ticker}/max-pain",
    "/api/stock/{ticker}/historical-risk-reversal-skew",
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

    for eid, ctx in contexts.items():
        payload = effective_payloads.get(eid)
        routing_key = EXTRACTOR_REGISTRY.get(ctx.path)
        
        safe_stale_age = ctx.stale_age_min if ctx.stale_age_min is not None else 999999
        
        if routing_key == "GEX":
            f_bundle = extract_gex_sign(payload, ctx)
            for k, v in f_bundle.features.items():
                candidates.append(FeatureCandidate(k, v, copy.deepcopy(f_bundle.meta.get("gex", {})), rank_freshness(ctx.freshness_state), safe_stale_age, PATH_PRIORITY.get(ctx.path, 99), eid, v is None))
            
            if ctx.freshness_state not in ("ERROR", "EMPTY_VALID") and payload and grab_list(payload):
                levels = build_gex_levels(payload)
                for l_type, price, mag, details in levels:
                    meta = _build_meta(ctx, "build_gex_levels", {"metric_name": "gex_levels", "fields_used": ["strike", "gamma_exposure"]}, details)
                    l_rows.append({"level_type": l_type, "price": price, "magnitude": mag, "meta_json": meta})
                
        elif routing_key == "FLOW":
            f_bundle = extract_smart_whale_pressure(payload, ctx)
            for k, v in f_bundle.features.items():
                candidates.append(FeatureCandidate(k, v, copy.deepcopy(f_bundle.meta.get("flow", {})), rank_freshness(ctx.freshness_state), safe_stale_age, PATH_PRIORITY.get(ctx.path, 99), eid, v is None))
                
        elif routing_key == "GREEKS":
            f_bundle = extract_dealer_greeks(payload, ctx)
            for k, v in f_bundle.features.items():
                candidates.append(FeatureCandidate(k, v, copy.deepcopy(f_bundle.meta.get("greeks", {})), rank_freshness(ctx.freshness_state), safe_stale_age, PATH_PRIORITY.get(ctx.path, 99), eid, v is None))
                
        elif routing_key == "PRICE":
            f_bundle = extract_price_features(payload, ctx)
            for k, v in f_bundle.features.items():
                candidates.append(FeatureCandidate(k, v, copy.deepcopy(f_bundle.meta.get("price", {})), rank_freshness(ctx.freshness_state), safe_stale_age, PATH_PRIORITY.get(ctx.path, 99), eid, v is None))
                
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

    return f_rows, l_rows