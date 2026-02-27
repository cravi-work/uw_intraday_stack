# src/analytics.py
import logging
from typing import Any, Dict, List, Optional, Tuple

# STRICT UNIDIRECTIONAL IMPORT: Only depends on neutral 'na' module. 
# NO imports from .features are present. _find_first is explicitly removed.
from .na import safe_float, grab_list

logger = logging.getLogger(__name__)

def build_gex_levels(payload: Any) -> List[Tuple[str, Optional[float], Optional[float], Dict[str, Any]]]:
    """
    Computes key gamma exposure (GEX) levels from spot-exposures payload.
    Returns: list of (level_type, price, magnitude, details)
    """
    rows = grab_list(payload)
    if not rows:
        return []
        
    parsed = []
    for r in rows:
        strike = safe_float(r.get("strike") or r.get("strike_price"))
        gex = safe_float(r.get("gamma_exposure") or r.get("gex") or r.get("gamma") or r.get("total_gamma"))
        if strike is not None and gex is not None:
            parsed.append({"strike": strike, "gex": gex})
            
    if not parsed:
        return []
        
    pos_sorted = sorted([p for p in parsed if p["gex"] > 0], key=lambda x: (-x["gex"], x["strike"]))
    neg_sorted = sorted([p for p in parsed if p["gex"] < 0], key=lambda x: (x["gex"], x["strike"]))
    
    levels = []
    if pos_sorted:
        levels.append(("GEX_POS_MAX", pos_sorted[0]["strike"], pos_sorted[0]["gex"], {
            "input_rows": len(rows), "parsed_rows": len(parsed)
        }))
        
    if neg_sorted:
        levels.append(("GEX_NEG_MAX", neg_sorted[0]["strike"], neg_sorted[0]["gex"], {
            "input_rows": len(rows), "parsed_rows": len(parsed)
        }))
        
    # Task 6: True Sign-Crossing Interpolation for GEX_FLIP
    strike_sorted = sorted(parsed, key=lambda x: x["strike"])
    crossings = []
    
    for i in range(len(strike_sorted) - 1):
        p1 = strike_sorted[i]
        p2 = strike_sorted[i+1]
        
        # Detect adjacent strikes where gamma exposure changes sign
        if p1["gex"] * p2["gex"] < 0:
            g_range = p2["gex"] - p1["gex"]
            s_range = p2["strike"] - p1["strike"]
            
            if g_range != 0:
                flip_strike = p1["strike"] - p1["gex"] * (s_range / g_range)
                crossings.append({
                    "flip_strike": flip_strike,
                    "gradient": abs(g_range),
                    "p1_strike": p1["strike"],
                    "p2_strike": p2["strike"]
                })
                
    if crossings:
        # Pick the crossing with the most significant gamma transition
        best_crossing = sorted(crossings, key=lambda x: (-x["gradient"], x["flip_strike"]))[0]
        levels.append(("GEX_FLIP", round(best_crossing["flip_strike"], 4), None, {
            "input_rows": len(rows), 
            "parsed_rows": len(parsed), 
            "method": "sign_crossing_interpolation",
            "p1_strike": best_crossing["p1_strike"],
            "p2_strike": best_crossing["p2_strike"]
        }))
        
    return levels

def build_oi_walls(payload: Any, spot: Optional[float] = None) -> List[Tuple[str, Optional[float], Optional[float], Dict[str, Any]]]:
    """
    Computes primary Option Interest (OI) support/resistance levels.
    """
    rows = grab_list(payload)
    if not rows:
        return []
    
    # Try to infer spot from rows if not provided explicitly
    if spot is None:
        for r in rows:
            s = safe_float(r.get("spot") or r.get("underlying_price") or r.get("underlying"))
            if s is not None:
                spot = s
                break
    
    parsed = []
    has_put_call = False
    
    for r in rows:
        strike = safe_float(r.get("strike") or r.get("strike_price"))
        oi = safe_float(r.get("open_interest") or r.get("oi"))
        pc_raw = r.get("put_call") or r.get("option_type") or r.get("type") or r.get("pc")
        
        if strike is not None and oi is not None:
            pc_norm = None
            if pc_raw:
                has_put_call = True
                pc_str = str(pc_raw).upper().strip()
                if pc_str in ("C", "CALL", "CALLS"):
                    pc_norm = "CALL"
                elif pc_str in ("P", "PUT", "PUTS"):
                    pc_norm = "PUT"
            
            parsed.append({"strike": strike, "oi": oi, "put_call": pc_norm})
            
    if not parsed:
        return []
        
    levels = []
    
    if not has_put_call:
        # Fallback: Generic Wall (No put_call available to separate direction)
        sorted_oi = sorted(parsed, key=lambda x: (-x["oi"], x["strike"]))
        levels.append(("OI_GENERIC_WALL", sorted_oi[0]["strike"], sorted_oi[0]["oi"], {
            "input_rows": len(rows), "parsed_rows": len(parsed), "degraded_reason": "missing_put_call"
        }))
        return levels
        
    if spot is None:
        # Fallback: Generic Wall (No spot available to establish above/below boundary)
        sorted_oi = sorted(parsed, key=lambda x: (-x["oi"], x["strike"]))
        levels.append(("OI_GENERIC_WALL", sorted_oi[0]["strike"], sorted_oi[0]["oi"], {
            "input_rows": len(rows), "parsed_rows": len(parsed), "degraded_reason": "missing_spot"
        }))
        return levels
        
    # Task 7: We have spot and put_call. Find Directional Walls.
    calls = [p for p in parsed if p["put_call"] == "CALL" and p["strike"] >= spot]
    puts = [p for p in parsed if p["put_call"] == "PUT" and p["strike"] <= spot]
    
    if calls:
        sorted_calls = sorted(calls, key=lambda x: (-x["oi"], x["strike"]))
        levels.append(("CALL_WALL", sorted_calls[0]["strike"], sorted_calls[0]["oi"], {
            "input_rows": len(rows), "parsed_rows": len(parsed), "spot_reference": spot
        }))
        
    if puts:
        sorted_puts = sorted(puts, key=lambda x: (-x["oi"], x["strike"]))
        levels.append(("PUT_WALL", sorted_puts[0]["strike"], sorted_puts[0]["oi"], {
            "input_rows": len(rows), "parsed_rows": len(parsed), "spot_reference": spot
        }))
        
    return levels

def build_darkpool_levels(payload: Any) -> List[Tuple[str, Optional[float], Optional[float], Dict[str, Any]]]:
    """
    Computes key darkpool block levels based on volume clustering.
    """
    rows = grab_list(payload)
    if not rows:
        return []
        
    parsed = []
    for r in rows:
        price = safe_float(r.get("price"))
        vol = safe_float(r.get("volume")) or safe_float(r.get("size"))
        if price is not None and vol is not None:
            parsed.append({"price": price, "vol": vol})
            
    if not parsed:
        return []
        
    # Aggregate volume by tight price node
    agg = {}
    for p in parsed:
        pr = round(p["price"], 2)
        agg[pr] = agg.get(pr, 0.0) + p["vol"]
        
    if not agg:
        return []
        
    sorted_levels = sorted(agg.items(), key=lambda x: (-x[1], x[0]))
    return [("DARKPOOL_MAX_VOL", sorted_levels[0][0], sorted_levels[0][1], {
        "input_rows": len(rows), "parsed_nodes": len(agg)
    })]

# Fixed: Changed "module" to "module_name" to prevent LogRecord KeyErrors
logger.info("Analytics module initialized successfully", extra={"event": "module_init", "module_name": "analytics"})