from __future__ import annotations
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from .na import safe_float, is_na, grab_list
from .endpoint_truth import EndpointContext

_grab_list = grab_list 
_as_float = safe_float 

@dataclass
class FeatureBundle:
    features: Dict[str, Optional[float]]
    meta: Dict[str, Any]

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
    """Constructs the standardized Meta JSON schema required for downstream auditing."""
    d = {"extractor": extractor_name}
    if details:
        d.update(details)
        
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
    """Overrides the metadata to explicitly indicate a deterministic failure state."""
    meta = _build_meta(ctx, extractor_name)
    meta["freshness_state"] = "ERROR"
    meta["na_reason"] = na_reason
    return meta

# --- Extractors ---

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


def extract_smart_whale_pressure(
    flow_payload: Any,
    ctx: EndpointContext,
    min_premium: float = 10000.0,
    max_dte: float = 14.0,
    norm_scale: float = 500_000.0
) -> FeatureBundle:
    
    if is_na(flow_payload) or ctx.freshness_state == "ERROR":
        return FeatureBundle({"smart_whale_pressure": None}, {"flow": _build_error_meta(ctx, "extract_smart_whale_pressure", ctx.na_reason or "missing_dependency_payload")})

    trades = grab_list(flow_payload)
    
    # Strict List Container Check
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
        # Case 1: Not a list at all
        if not is_list_container:
            return FeatureBundle({"smart_whale_pressure": None}, {"flow": _build_error_meta(ctx, "extract_smart_whale_pressure", "unrecognized_schema")})
            
        # Case 2: List exists but items are not dicts
        if raw_list and len(raw_list) > 0:
             return FeatureBundle({"smart_whale_pressure": None}, {"flow": _build_error_meta(ctx, "extract_smart_whale_pressure", "schema_non_dict_rows")})
             
        # Case 3: Valid Empty List -> 0.0 (No Volume, safe empty state)
        meta = _build_meta(ctx, "extract_smart_whale_pressure", {"status": "computed_zero_from_empty_valid", "n_trades": 0})
        return FeatureBundle({"smart_whale_pressure": 0.0}, {"flow": meta})

    # --- Processing ---
    whale_call = 0.0
    whale_put = 0.0
    
    # Counters
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
        
        # 1. Integrity Check
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

        # 2. Logic Filters (Policy)
        if prem < min_premium or dte > max_dte:
            skip_threshold += 1
            continue
            
        # 3. Aggregation
        valid_count += 1
        if is_bull:
            if pc == "CALL": whale_call += prem
            else: whale_put += prem
        elif is_bear:
            if pc == "CALL": whale_call -= prem
            else: whale_put -= prem

    # --- Final Verdict ---
    if valid_count == 0:
        if parseable_count > 0:
            meta = _build_meta(ctx, "extract_smart_whale_pressure", {
                "status": "filtered_zero", 
                "n_raw_trades": len(trades), 
                "parseable": parseable_count,
                "skipped_threshold": skip_threshold,
                "policy": {"min_prem": min_premium, "max_dte": max_dte}
            })
            return FeatureBundle({"smart_whale_pressure": 0.0}, {"flow": meta})
            
        if skip_missing_fields > 0:
             return FeatureBundle({"smart_whale_pressure": None}, {"flow": _build_error_meta(ctx, "extract_smart_whale_pressure", "missing_required_fields")})
        if skip_bad_type > 0:
             return FeatureBundle({"smart_whale_pressure": None}, {"flow": _build_error_meta(ctx, "extract_smart_whale_pressure", "unrecognized_put_call")})
        if skip_bad_side > 0:
             return FeatureBundle({"smart_whale_pressure": None}, {"flow": _build_error_meta(ctx, "extract_smart_whale_pressure", "unrecognized_side_labels")})
             
        return FeatureBundle({"smart_whale_pressure": None}, {"flow": _build_error_meta(ctx, "extract_smart_whale_pressure", "no_valid_trades_unknown")})

    net = whale_call - whale_put
    meta = _build_meta(ctx, "extract_smart_whale_pressure", {"net_prem": net, "n_valid": valid_count, "n_raw": len(trades)})
    
    return FeatureBundle({"smart_whale_pressure": _normalize_signed(net, scale=norm_scale)}, {"flow": meta})


def extract_dealer_greeks(
    greek_payload: Any,
    ctx: EndpointContext,
    norm_scale: float = 1_000_000_000.0
) -> FeatureBundle:
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
    
    if abs(tot_gamma) <= 1e-9:
        sign = 0.0
    else:
        sign = 1.0 if tot_gamma > 0 else -1.0
        
    meta = _build_meta(ctx, "extract_gex_sign", {"total": tot_gamma, "n_strikes": valid_rows})
    return FeatureBundle({"net_gex_sign": sign}, {"gex": meta})

# --- Consistent Stubs ---
def extract_strike_flow_pressure(payload_a: Any, payload_b: Any, ctx: EndpointContext) -> FeatureBundle: 
    return FeatureBundle({"strike_flow_imbalance": None}, {"strike_flow": _build_error_meta(ctx, "extract_strike_flow_pressure", "not_implemented")})

def extract_oi_shifts(payload: Any, ctx: EndpointContext) -> FeatureBundle: 
    return FeatureBundle({"net_oi_change": None}, {"oi": _build_error_meta(ctx, "extract_oi_shifts", "not_implemented")})

def extract_term_structure(payload: Any, ctx: EndpointContext) -> FeatureBundle: 
    return FeatureBundle({"term_structure_slope": None}, {"term_structure": _build_error_meta(ctx, "extract_term_structure", "not_implemented")})

def extract_lit_vs_dp_divergence(payload_a: Any, payload_b: Any, ctx: EndpointContext) -> FeatureBundle: 
    return FeatureBundle({"lit_vs_dp_divergence": None}, {"lit_dp": _build_error_meta(ctx, "extract_lit_vs_dp_divergence", "not_implemented")})

def extract_volatility_regime(payload: Any, ctx: EndpointContext) -> FeatureBundle: 
    return FeatureBundle({"iv_rank_regime": None}, {"vol": _build_error_meta(ctx, "extract_volatility_regime", "not_implemented")})

def extract_gamma_squeeze_fuel(payload: Any, ctx: EndpointContext) -> FeatureBundle: 
    return FeatureBundle({"gamma_squeeze_fuel": None}, {"squeeze": _build_error_meta(ctx, "extract_gamma_squeeze_fuel", "not_implemented")})