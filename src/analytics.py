import logging
from typing import Any, Dict, List, Optional, Tuple
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
        
    # Tie-breakers: rank by highest absolute gex, then lowest strike to ensure 100% deterministic output
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
        
    # Find GEX flip (strike closest to 0 GEX crossing)
    abs_sorted = sorted(parsed, key=lambda x: (abs(x["gex"]), x["strike"]))
    if abs_sorted:
        # Strict NA discipline: magnitude is None (not 0.0) since it is a cross-level, not a magnitude level
        levels.append(("GEX_FLIP", abs_sorted[0]["strike"], None, {
            "input_rows": len(rows), "parsed_rows": len(parsed), "method": "min_abs_gex"
        }))
        
    return levels

logger.info("Analytics module initialized successfully", extra={"event": "module_init", "module": "analytics"})