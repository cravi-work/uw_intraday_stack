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
        
    abs_sorted = sorted(parsed, key=lambda x: (abs(x["gex"]), x["strike"]))
    if abs_sorted:
        levels.append(("GEX_FLIP", abs_sorted[0]["strike"], None, {
            "input_rows": len(rows), "parsed_rows": len(parsed), "method": "min_abs_gex"
        }))
        
    return levels

def build_oi_walls(payload: Any) -> List[Tuple[str, Optional[float], Optional[float], Dict[str, Any]]]:
    """
    Computes primary Option Interest (OI) support/resistance levels.
    """
    rows = grab_list(payload)
    if not rows:
        return []
    
    parsed = []
    for r in rows:
        strike = safe_float(r.get("strike") or r.get("strike_price"))
        oi = safe_float(r.get("open_interest") or r.get("oi"))
        if strike is not None and oi is not None:
            parsed.append({"strike": strike, "oi": oi})
            
    if not parsed:
        return []
        
    # Find max absolute OI node
    sorted_oi = sorted(parsed, key=lambda x: (-x["oi"], x["strike"]))
    return [("OI_MAX_WALL", sorted_oi[0]["strike"], sorted_oi[0]["oi"], {
        "input_rows": len(rows), "parsed_rows": len(parsed)
    })]

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

logger.info("Analytics module initialized successfully", extra={"event": "module_init", "module": "analytics"})