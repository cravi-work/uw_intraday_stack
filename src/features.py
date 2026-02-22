from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Mapping
from dataclasses import dataclass
import math
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

def _find_first(obj: Any, keys: List[str]) -> Any:
    if not isinstance(obj, dict): return None
    for k in keys:
        if k in obj: return obj[k]
    return None

def _normalize_signed(x: Optional[float], *, scale: float) -> Optional[float]:
    if scale == 0: return None 
    val = safe_float(x)
    if val is None: return None
    return max(-1.0, min(1.0, val / scale))

def merge_bundles(bundles: List[FeatureBundle]) -> FeatureBundle:
    f_out = {}
    m_out = {}
    for b in bundles:
        f_out.update(b.features)
        for ns_key, ns_val in b.meta.items():
            if ns_key in m_out:
                raise ValueError(f"Meta namespace collision detected: '{ns_key}'")
            m_out[ns_key] = ns_val
    return FeatureBundle(f_out, m_out)

def _build_meta(ctx: EndpointContext, extractor_name: str, details: Dict[str, Any] = None) -> Dict[str, Any]:
    d = {"extractor": extractor_name}
    if details: d.update(details)
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
        "details": d
    }

def _build_error_meta(ctx: EndpointContext, extractor_name: str, na_reason: str) -> Dict[str, Any]:
    meta = _build_meta(ctx, extractor_name)
    meta["freshness_state"] = "ERROR"
    meta["na_reason"] = na_reason
    return meta

def extract_price_features(ohlc_payload: Any, ctx: EndpointContext) -> FeatureBundle:
    if is_na(ohlc_payload) or ctx.freshness_state == "ERROR":
        return FeatureBundle({"spot": None}, {"price": _build_error_meta(ctx, "extract_price_features", ctx.na_reason or "missing_dependency_payload")})
        
    rows = grab_list(ohlc_payload)
    if not rows:
        return FeatureBundle({"spot": None}, {"price": _build_error_meta(ctx, "extract_price_features", "no_rows")})
    
    last = rows[-1]
    close = safe_float(_find_first(last, ["close", "c", "price"]))
    ts = _find_first(last, ["t", "ts", "date"])
    
    if close is None:
        return FeatureBundle({"spot": None}, {"price": _build_error_meta(ctx, "extract_price_features", "missing_close_field")})
        
    return FeatureBundle({"spot": close}, {"price": _build_meta(ctx, "extract_price_features", {"last_ts": ts})})

def extract_smart_whale_pressure(flow_payload: Any, ctx: EndpointContext, min_premium: float = 10000.0, max_dte: float = 14.0, norm_scale: float = 500_000.0) -> FeatureBundle:
    if is_na(flow_payload) or ctx.freshness_state == "ERROR":
        return FeatureBundle({"smart_whale_pressure": None}, {"flow": _build_error_meta(ctx, "extract_smart_whale_pressure", ctx.na_reason or "missing_dependency_payload")})

    trades = grab_list(flow_payload)
    raw_list = None
    is_list_container = False
    
    if isinstance(flow_payload, list):
        is_list_container = True
        raw_list = flow_payload
    elif isinstance(flow_payload, dict):
        for k in ["data", "trades", "results", "history", "items"]:
            val = flow_payload.get(k)
            if val is not None and isinstance(val, list):
                is_list_container = True
                raw_list = val
                break
    
    if not trades:
        if not is_list_container:
            return FeatureBundle({"smart_whale_pressure": None}, {"flow": _build_error_meta(ctx, "extract_smart_whale_pressure", "unrecognized_schema")})
        if raw_list and len(raw_list) > 0:
             return FeatureBundle({"smart_whale_pressure": None}, {"flow": _build_error_meta(ctx, "extract_smart_whale_pressure", "schema_non_dict_rows")})
        meta = _build_meta(ctx, "extract_smart_whale_pressure", {"status": "computed_zero_from_empty_valid", "n_trades": 0})
        return FeatureBundle({"smart_whale_pressure": 0.0}, {"flow": meta})

    whale_call = 0.0
    whale_put = 0.0
    valid_count = 0
    parseable_count = 0
    skip_missing_fields = 0
    skip_bad_type = 0
    skip_bad_side = 0
    skip_threshold = 0
    
    for t in trades:
        prem = safe_float(_find_first(t, ["premium", "total_premium", "cost", "value"]))
        dte = safe_float(_find_first(t, ["dte", "exp_days"]))
        side_raw = _find_first(t, ["side", "sentiment", "type"])
        pc_raw = _find_first(t, ["put_call", "option_type", "right"])
        
        if prem is None or dte is None or is_na(side_raw) or is_na(pc_raw):
            skip_missing_fields += 1
            continue
            
        pc = str(pc_raw).upper()
        if pc not in ["CALL", "PUT"]:
            skip_bad_type += 1
            continue
            
        side = str(side_raw).upper()
        is_bull = side in ("ASK", "BUY", "BULLISH")
        is_bear = side in ("BID", "SELL", "BEARISH")
        
        if not (is_bull or is_bear):
            skip_bad_side += 1 
            continue

        parseable_count += 1

        if prem < min_premium or dte > max_dte:
            skip_threshold += 1
            continue
            
        valid_count += 1
        if is_bull:
            if pc == "CALL": whale_call += prem
            else: whale_put += prem
        elif is_bear:
            if pc == "CALL": whale_call -= prem
            else: whale_put -= prem

    if valid_count == 0:
        if parseable_count > 0:
            meta = _build_meta(ctx, "extract_smart_whale_pressure", {
                "status": "filtered_zero", 
                "n_raw_trades": len(trades), "parseable": parseable_count,
                "skipped_threshold": skip_threshold, "policy": {"min_prem": min_premium, "max_dte": max_dte}
            })
            return FeatureBundle({"smart_whale_pressure": 0.0}, {"flow": meta})
            
        if skip_missing_fields > 0: return FeatureBundle({"smart_whale_pressure": None}, {"flow": _build_error_meta(ctx, "extract_smart_whale_pressure", "missing_required_fields")})
        if skip_bad_type > 0: return FeatureBundle({"smart_whale_pressure": None}, {"flow": _build_error_meta(ctx, "extract_smart_whale_pressure", "unrecognized_put_call")})
        if skip_bad_side > 0: return FeatureBundle({"smart_whale_pressure": None}, {"flow": _build_error_meta(ctx, "extract_smart_whale_pressure", "unrecognized_side_labels")})
        return FeatureBundle({"smart_whale_pressure": None}, {"flow": _build_error_meta(ctx, "extract_smart_whale_pressure", "no_valid_trades_unknown")})

    net = whale_call - whale_put
    meta = _build_meta(ctx, "extract_smart_whale_pressure", {"net_prem": net, "n_valid": valid_count, "n_raw": len(trades)})
    return FeatureBundle({"smart_whale_pressure": _normalize_signed(net, scale=norm_scale)}, {"flow": meta})

def extract_dealer_greeks(greek_payload: Any, ctx: EndpointContext, norm_scale: float = 1_000_000_000.0) -> FeatureBundle:
    keys = ["dealer_vanna", "dealer_charm", "net_gamma_notional"]
    if is_na(greek_payload) or ctx.freshness_state == "ERROR":
        return FeatureBundle({k: None for k in keys}, {"greeks": _build_error_meta(ctx, "extract_dealer_greeks", ctx.na_reason or "missing_dependency_payload")})

    rows = grab_list(greek_payload)
    if not rows:
        return FeatureBundle({k: None for k in keys}, {"greeks": _build_error_meta(ctx, "extract_dealer_greeks", "no_rows")})

    latest = rows[-1]
    def _sum(row, metric):
        t = safe_float(_find_first(row, [metric, f"total_{metric}", f"{metric}_exposure"]))
        if t is not None: return t
        c = safe_float(_find_first(row, [f"call_{metric}", f"calls_{metric}"]))
        p = safe_float(_find_first(row, [f"put_{metric}", f"puts_{metric}"]))
        if c is not None and p is not None: return c + p
        return None

    meta = _build_meta(ctx, "extract_dealer_greeks", {"ts": latest.get("date"), "scale_used": norm_scale})
    return FeatureBundle({
        "dealer_vanna": _normalize_signed(_sum(latest, "vanna"), scale=norm_scale),
        "dealer_charm": _normalize_signed(_sum(latest, "charm"), scale=norm_scale),
        "net_gamma_notional": _normalize_signed(_sum(latest, "gamma"), scale=norm_scale)
    }, {"greeks": meta})

def extract_gex_sign(spot_exposures_payload: Any, ctx: EndpointContext) -> FeatureBundle:
    if is_na(spot_exposures_payload) or ctx.freshness_state == "ERROR":
        return FeatureBundle({"net_gex_sign": None}, {"gex": _build_error_meta(ctx, "extract_gex_sign", ctx.na_reason or "missing_dependency_payload")})

    rows = grab_list(spot_exposures_payload)
    if not rows:
        return FeatureBundle({"net_gex_sign": None}, {"gex": _build_error_meta(ctx, "extract_gex_sign", "no_rows")})
    
    tot_gamma = 0.0
    valid_rows = 0
    for r in rows:
        g = safe_float(_find_first(r, ["gamma_exposure", "gex", "gamma", "exposure"]))
        if g is not None:
            tot_gamma += g
            valid_rows += 1
            
    if valid_rows == 0:
        return FeatureBundle({"net_gex_sign": None}, {"gex": _build_error_meta(ctx, "extract_gex_sign", "missing_gamma_fields")})
    
    if abs(tot_gamma) <= 1e-9: sign = 0.0
    else: sign = 1.0 if tot_gamma > 0 else -1.0
        
    meta = _build_meta(ctx, "extract_gex_sign", {"total": tot_gamma, "n_strikes": valid_rows})
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
        
        if routing_key == "GEX":
            f_bundle = extract_gex_sign(payload, ctx)
            for k, v in f_bundle.features.items():
                candidates.append(FeatureCandidate(k, v, f_bundle.meta.get("gex", {}), rank_freshness(ctx.freshness_state), ctx.stale_age_min or 999999, PATH_PRIORITY.get(ctx.path, 99), eid, v is None))
            levels = build_gex_levels(payload)
            for l_type, price, mag, details in levels:
                l_rows.append({"level_type": l_type, "price": price, "magnitude": mag, "meta_json": _build_meta(ctx, "build_gex_levels", details)})
                
        elif routing_key == "FLOW":
            f_bundle = extract_smart_whale_pressure(payload, ctx)
            for k, v in f_bundle.features.items():
                candidates.append(FeatureCandidate(k, v, f_bundle.meta.get("flow", {}), rank_freshness(ctx.freshness_state), ctx.stale_age_min or 999999, PATH_PRIORITY.get(ctx.path, 99), eid, v is None))
                
        elif routing_key == "GREEKS":
            f_bundle = extract_dealer_greeks(payload, ctx)
            for k, v in f_bundle.features.items():
                candidates.append(FeatureCandidate(k, v, f_bundle.meta.get("greeks", {}), rank_freshness(ctx.freshness_state), ctx.stale_age_min or 999999, PATH_PRIORITY.get(ctx.path, 99), eid, v is None))
                
        elif routing_key == "PRICE":
            f_bundle = extract_price_features(payload, ctx)
            for k, v in f_bundle.features.items():
                candidates.append(FeatureCandidate(k, v, f_bundle.meta.get("price", {}), rank_freshness(ctx.freshness_state), ctx.stale_age_min or 999999, PATH_PRIORITY.get(ctx.path, 99), eid, v is None))
                
        elif ctx.path not in PRESENCE_ONLY_ENDPOINTS:
            raise RuntimeError(f"CRITICAL EXTRACTOR COVERAGE GAP: Endpoint path '{ctx.path}' is not mapped in EXTRACTOR_REGISTRY and not whitelisted in PRESENCE_ONLY_ENDPOINTS.")

    # Deduplicate candidates using institutional strict ranking
    grouped: Dict[str, List[FeatureCandidate]] = {}
    for c in candidates: grouped.setdefault(c.feature_key, []).append(c)

    f_rows: List[FeatureRow] = []
    for f_key, group in grouped.items():
        group.sort(key=lambda x: (x.is_none, x.freshness_rank, x.stale_age, x.path_priority, x.endpoint_id))
        
        best = group[0]
        
        # Conflict detection: same rank, different non-None values
        for other in group[1:]:
            if (other.is_none == best.is_none and other.freshness_rank == best.freshness_rank and 
                other.stale_age == best.stale_age and other.path_priority == best.path_priority):
                if best.feature_value is not None and other.feature_value is not None:
                    if not math.isclose(best.feature_value, other.feature_value, abs_tol=1e-9):
                        raise RuntimeError(f"FEATURE_CONFLICT:{f_key} - Endpoint {best.endpoint_id} vs {other.endpoint_id} generated divergent values at equal rank.")
                    
        meta = best.meta_json
        if len(group) > 1:
            meta.setdefault("details", {})["shadowed_candidates"] = [
                {"endpoint_id": c.endpoint_id, "is_none": c.is_none, "freshness_rank": c.freshness_rank} for c in group[1:]
            ]
            
        f_rows.append({"feature_key": best.feature_key, "feature_value": best.feature_value, "meta_json": meta})

    return f_rows, l_rows