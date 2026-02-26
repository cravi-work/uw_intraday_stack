from __future__ import annotations
import copy
import datetime
import logging
import math
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Mapping
from dataclasses import dataclass

from .na import safe_float, is_na, grab_list
from .endpoint_truth import EndpointContext
from .analytics import build_gex_levels, build_oi_walls, build_darkpool_levels

logger = logging.getLogger(__name__)

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
    "/api/stock/{ticker}/ohlc/{candle_size}": 1,
    "/api/darkpool/{ticker}": 1,
    "/api/lit-flow/{ticker}": 1,
    "/api/stock/{ticker}/volatility/term-structure": 1,
    "/api/stock/{ticker}/historical-risk-reversal-skew": 1
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

def _build_meta(
    ctx: EndpointContext, 
    extractor_name: str, 
    lineage: Dict[str, Any], 
    details: Dict[str, Any] = None
) -> Dict[str, Any]:
    
    d = {"extractor": extractor_name}
    if details: 
        d.update(details)
        
    eff_ts = None
    ts_source = "missing"
    ts_quality = "MISSING"
    
    if details and "effective_ts_utc" in details:
        payload_ts = details.get("effective_ts_utc")
        if payload_ts in (None, "INVALID"):
            ts_source = "payload"
            ts_quality = "INVALID"
            eff_ts = None
        else:
            ts_source = "payload"
            ts_quality = "VALID"
            eff_ts = payload_ts
    elif getattr(ctx, "effective_ts_utc", None):
        ts_source = "endpoint_context"
        ts_quality = "VALID"
        eff_ts = ctx.effective_ts_utc.isoformat()
        
    full_lineage = {
        "metric_name": lineage.get("metric_name", "unknown"),
        "source_path": ctx.path,
        "fields_used": lineage.get("fields_used", []),
        "units_expected": lineage.get("units_expected", "unknown"),
        "normalization": lineage.get("normalization", "none"),
        "session_applicability": lineage.get("session_applicability", "PREMARKET/RTH/AFTERHOURS"),
        "quality_policy": lineage.get("quality_policy", "None on missing"),
        "criticality": lineage.get("criticality", "NON_CRITICAL"),
        "effective_ts_utc": eff_ts,
        "timestamp_source": ts_source,
        "timestamp_quality": ts_quality
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

def extract_price_features(ohlc_payload: Any, ctx: EndpointContext) -> FeatureBundle:
    lineage = {
        "metric_name": "spot",
        "fields_used": ["close", "t"],
        "units_expected": "USD",
        "normalization": "none",
        "session_applicability": "PREMARKET/RTH/AFTERHOURS",
        "quality_policy": "None if missing required explicit keys",
        "criticality": "CRITICAL"
    }
    
    if is_na(ohlc_payload) or ctx.freshness_state == "ERROR":
        return FeatureBundle({"spot": None}, {"price": _build_error_meta(ctx, "extract_price_features", lineage, ctx.na_reason or "missing_dependency")})
        
    rows = grab_list(ohlc_payload)
    if not rows:
        return FeatureBundle({"spot": None}, {"price": _build_error_meta(ctx, "extract_price_features", lineage, "no_rows")})
    
    latest_row = max(rows, key=lambda r: _parse_strict_ts(r, "t"))
    
    close_val = latest_row.get("close")
    t_val = latest_row.get("t")
    
    if close_val is None:
        return FeatureBundle({"spot": None}, {"price": _build_error_meta(ctx, "extract_price_features", lineage, "missing_explicit_close_field")})
        
    close_float = safe_float(close_val)
    ts_float = _parse_strict_ts(latest_row, "t")
    
    eff_ts = datetime.datetime.fromtimestamp(ts_float, datetime.timezone.utc).isoformat() if ts_float > 0 else "INVALID"
    
    return FeatureBundle({"spot": close_float}, {"price": _build_meta(ctx, "extract_price_features", lineage, {"last_ts": t_val, "effective_ts_utc": eff_ts})})

def extract_smart_whale_pressure(flow_payload: Any, ctx: EndpointContext, min_premium: float = 10000.0, max_dte: float = 14.0, norm_scale: float = 500_000.0) -> FeatureBundle:
    lineage = {
        "metric_name": "smart_whale_pressure",
        "fields_used": ["premium", "dte", "side", "put_call"],
        "units_expected": "Net Premium Flow (USD)",
        "normalization": f"normalize_signed [-1, 1] by {norm_scale}",
        "session_applicability": "RTH",
        "quality_policy": "None on filtered zeros to avoid false baseline certainty",
        "criticality": "CRITICAL"
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
        "session_applicability": "PREMARKET/RTH",
        "quality_policy": "None on missing",
        "criticality": "CRITICAL"
    }
    
    if is_na(greek_payload) or ctx.freshness_state == "ERROR":
        return FeatureBundle({k: None for k in keys}, {"greeks": _build_error_meta(ctx, "extract_dealer_greeks", lineage, ctx.na_reason or "missing_dependency")})

    rows = grab_list(greek_payload)
    if not rows:
        return FeatureBundle({k: None for k in keys}, {"greeks": _build_error_meta(ctx, "extract_dealer_greeks", lineage, "no_rows")})

    latest = max(rows, key=lambda r: _parse_strict_ts(r, "date"))
    
    ts_float = _parse_strict_ts(latest, "date")
    eff_ts = datetime.datetime.fromtimestamp(ts_float, datetime.timezone.utc).isoformat() if ts_float > 0 else "INVALID"
    
    meta = _build_meta(ctx, "extract_dealer_greeks", lineage, {"ts": latest.get("date"), "scale_used": norm_scale, "effective_ts_utc": eff_ts})
    
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
        "session_applicability": "PREMARKET/RTH",
        "quality_policy": "None on missing exposure fields",
        "criticality": "CRITICAL"
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

def extract_oi_features(payload: Any, ctx: EndpointContext) -> FeatureBundle:
    lineage = {
        "metric_name": "oi_pressure",
        "fields_used": ["open_interest", "strike"],
        "units_expected": "Contracts",
        "normalization": "none",
        "session_applicability": "RTH",
        "quality_policy": "None on missing",
        "criticality": "CRITICAL"
    }
    if is_na(payload) or ctx.freshness_state == "ERROR":
        return FeatureBundle({"oi_pressure": None}, {"oi": _build_error_meta(ctx, "extract_oi", lineage, ctx.na_reason or "missing_dependency")})
    
    rows = grab_list(payload)
    if not rows:
        return FeatureBundle({"oi_pressure": None}, {"oi": _build_error_meta(ctx, "extract_oi", lineage, "no_rows")})
        
    total_oi = 0.0
    for r in rows:
        val = safe_float(r.get("open_interest", 0.0))
        if val is not None and math.isfinite(val):
            total_oi += val
            
    return FeatureBundle({"oi_pressure": total_oi}, {"oi": _build_meta(ctx, "extract_oi", lineage, {"n_rows": len(rows)})})

def extract_volatility_features(payload: Any, ctx: EndpointContext) -> FeatureBundle:
    lineage = {
        "metric_name": "iv_rank",
        "fields_used": ["iv_rank", "iv_percentile"],
        "units_expected": "Percentile [0,1]",
        "normalization": "none",
        "session_applicability": "PREMARKET/RTH/AFTERHOURS",
        "quality_policy": "None on missing",
        "criticality": "NON_CRITICAL"
    }
    if is_na(payload) or ctx.freshness_state == "ERROR":
        return FeatureBundle({"iv_rank": None}, {"vol": _build_error_meta(ctx, "extract_vol", lineage, ctx.na_reason or "missing_dependency")})
    
    rows = grab_list(payload)
    if not rows and isinstance(payload, dict):
        rows = [payload]
        
    val = safe_float(rows[0].get("iv_rank")) if rows else None
    if val is None or not math.isfinite(val):
        return FeatureBundle({"iv_rank": None}, {"vol": _build_error_meta(ctx, "extract_vol", lineage, "missing_iv_rank")})
    
    return FeatureBundle({"iv_rank": val}, {"vol": _build_meta(ctx, "extract_vol", lineage, {})})

def extract_vol_term_structure(payload: Any, ctx: EndpointContext) -> FeatureBundle:
    lineage = {
        "metric_name": "vol_term_slope",
        "fields_used": ["dte", "days", "iv", "implied_volatility"],
        "units_expected": "IV Spread",
        "normalization": "none",
        "session_applicability": "PREMARKET/RTH/AFTERHOURS",
        "quality_policy": "None on missing",
        "criticality": "NON_CRITICAL"
    }
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
    
    return FeatureBundle({"vol_term_slope": slope}, {"vol_ts": _build_meta(ctx, "extract_vol_term_structure", lineage, {"n_rows": len(valid_pts)})})

def extract_vol_skew(payload: Any, ctx: EndpointContext) -> FeatureBundle:
    lineage = {
        "metric_name": "vol_skew",
        "fields_used": ["skew", "risk_reversal", "value"],
        "units_expected": "Skew Ratio",
        "normalization": "none",
        "session_applicability": "PREMARKET/RTH/AFTERHOURS",
        "quality_policy": "None on missing",
        "criticality": "NON_CRITICAL"
    }
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
        
    return FeatureBundle({"vol_skew": skew_val}, {"skew": _build_meta(ctx, "extract_vol_skew", lineage, {})})

def extract_darkpool_pressure(payload: Any, ctx: EndpointContext) -> FeatureBundle:
    lineage = {
        "metric_name": "darkpool_pressure",
        "fields_used": ["volume", "price", "size"],
        "units_expected": "Total Notional USD",
        "normalization": "none",
        "session_applicability": "PREMARKET/RTH/AFTERHOURS",
        "quality_policy": "None on missing",
        "criticality": "NON_CRITICAL"
    }
    if is_na(payload) or ctx.freshness_state == "ERROR":
        return FeatureBundle({"darkpool_pressure": None}, {"darkpool": _build_error_meta(ctx, "extract_darkpool", lineage, ctx.na_reason or "missing_dependency")})
    
    rows = grab_list(payload)
    if not rows:
        return FeatureBundle({"darkpool_pressure": None}, {"darkpool": _build_error_meta(ctx, "extract_darkpool", lineage, "no_rows")})
        
    total_notional = 0.0
    for r in rows:
        vol = safe_float(r.get("volume", 0)) or safe_float(r.get("size", 0)) or 0.0
        price = safe_float(r.get("price", 0)) or 0.0
        notional = vol * price
        if math.isfinite(notional):
            total_notional += notional
            
    return FeatureBundle({"darkpool_pressure": total_notional}, {"darkpool": _build_meta(ctx, "extract_darkpool", lineage, {"n_rows": len(rows)})})

def extract_litflow_pressure(payload: Any, ctx: EndpointContext) -> FeatureBundle:
    lineage = {
        "metric_name": "litflow_pressure",
        "fields_used": ["volume", "price", "side", "size"],
        "units_expected": "Net Notional USD",
        "normalization": "none",
        "session_applicability": "RTH",
        "quality_policy": "None on missing",
        "criticality": "NON_CRITICAL"
    }
    if is_na(payload) or ctx.freshness_state == "ERROR":
        return FeatureBundle({"litflow_pressure": None}, {"litflow": _build_error_meta(ctx, "extract_litflow", lineage, ctx.na_reason or "missing_dependency")})
    
    rows = grab_list(payload)
    if not rows:
        return FeatureBundle({"litflow_pressure": None}, {"litflow": _build_error_meta(ctx, "extract_litflow", lineage, "no_rows")})
        
    net_notional = 0.0
    for r in rows:
        vol = safe_float(r.get("volume", 0)) or safe_float(r.get("size", 0)) or 0.0
        price = safe_float(r.get("price", 0)) or 0.0
        side = str(r.get("side", "")).upper()
        
        notional = vol * price
        if not math.isfinite(notional):
            continue
            
        if side in ("ASK", "BUY", "BULL", "BULLISH"):
            net_notional += notional
        elif side in ("BID", "SELL", "BEAR", "BEARISH"):
            net_notional -= notional
            
    return FeatureBundle({"litflow_pressure": net_notional}, {"litflow": _build_meta(ctx, "extract_litflow", lineage, {"n_rows": len(rows)})})

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
                
        elif routing_key == "OI":
            f_bundle = extract_oi_features(payload, ctx)
            for k, v in f_bundle.features.items():
                candidates.append(FeatureCandidate(k, v, copy.deepcopy(f_bundle.meta.get("oi", {})), rank_freshness(ctx.freshness_state), safe_stale_age, PATH_PRIORITY.get(ctx.path, 99), eid, v is None))
            if ctx.freshness_state not in ("ERROR", "EMPTY_VALID") and payload and grab_list(payload):
                levels = build_oi_walls(payload)
                for l_type, price, mag, details in levels:
                    meta = _build_meta(ctx, "build_oi_walls", {"metric_name": "oi_walls", "fields_used": ["strike", "open_interest"]}, details)
                    l_rows.append({"level_type": l_type, "price": price, "magnitude": mag, "meta_json": meta})
                
        elif routing_key == "VOL":
            f_bundle = extract_volatility_features(payload, ctx)
            for k, v in f_bundle.features.items():
                candidates.append(FeatureCandidate(k, v, copy.deepcopy(f_bundle.meta.get("vol", {})), rank_freshness(ctx.freshness_state), safe_stale_age, PATH_PRIORITY.get(ctx.path, 99), eid, v is None))

        elif routing_key == "VOL_TERM":
            f_bundle = extract_vol_term_structure(payload, ctx)
            for k, v in f_bundle.features.items():
                candidates.append(FeatureCandidate(k, v, copy.deepcopy(f_bundle.meta.get("vol_ts", {})), rank_freshness(ctx.freshness_state), safe_stale_age, PATH_PRIORITY.get(ctx.path, 99), eid, v is None))

        elif routing_key == "VOL_SKEW":
            f_bundle = extract_vol_skew(payload, ctx)
            for k, v in f_bundle.features.items():
                candidates.append(FeatureCandidate(k, v, copy.deepcopy(f_bundle.meta.get("skew", {})), rank_freshness(ctx.freshness_state), safe_stale_age, PATH_PRIORITY.get(ctx.path, 99), eid, v is None))

        elif routing_key == "DARKPOOL":
            f_bundle = extract_darkpool_pressure(payload, ctx)
            for k, v in f_bundle.features.items():
                candidates.append(FeatureCandidate(k, v, copy.deepcopy(f_bundle.meta.get("darkpool", {})), rank_freshness(ctx.freshness_state), safe_stale_age, PATH_PRIORITY.get(ctx.path, 99), eid, v is None))
            if ctx.freshness_state not in ("ERROR", "EMPTY_VALID") and payload and grab_list(payload):
                levels = build_darkpool_levels(payload)
                for l_type, price, mag, details in levels:
                    meta = _build_meta(ctx, "build_darkpool_levels", {"metric_name": "darkpool_levels", "fields_used": ["price", "volume"]}, details)
                    l_rows.append({"level_type": l_type, "price": price, "magnitude": mag, "meta_json": meta})

        elif routing_key == "LITFLOW":
            f_bundle = extract_litflow_pressure(payload, ctx)
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

logger.info("Features module initialized successfully", extra={"event": "module_init", "module": "features"})