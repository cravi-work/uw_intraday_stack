"""
src/na.py
Centralized handling for 'Not Available' (NA) values.
Eliminates silent hallucinations by treating missing data as None, not 0.0.
"""
from typing import Any, Optional, TypeVar, List, Dict
import math

T = TypeVar("T")

# Expanded set including dashboard placeholders
NA_STRINGS = {
    "", " ", "\t", 
    "null", "none", "n/a", "na", "nan", "undefined",
    "missing", "unknown", "nil", "-", "—"
}

def is_na(val: Any) -> bool:
    """True if val is None, NaN, empty, or a known placeholder string."""
    if val is None:
        return True
    if isinstance(val, float):
        return math.isnan(val) or math.isinf(val)
    if isinstance(val, str):
        return val.strip().lower() in NA_STRINGS
    return False

def safe_float(val: Any) -> Optional[float]:
    """Converts to float, strictly returning None on failure/NA."""
    if is_na(val): return None
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f): return None
        return f
    except (ValueError, TypeError):
        return None

def grab_list(obj: Any) -> List[Dict[str, Any]]:
    """
    Robustly unwraps API lists.
    Guarantees output is List[Dict]. Filters out non-dict items.

    Supports the common vendor wrapper shapes used in this repo without
    attempting arbitrary schema flattening.
    """
    if is_na(obj):
        return []

    def _filter_dicts(lst: List[Any]) -> List[Dict[str, Any]]:
        return [x for x in lst if isinstance(x, dict)]

    def _unwrap(candidate: Any, depth: int = 0) -> List[Dict[str, Any]]:
        if depth > 4 or is_na(candidate):
            return []
        if isinstance(candidate, list):
            return _filter_dicts(candidate)
        if not isinstance(candidate, dict):
            return []

        for k in ["data", "trades", "results", "result", "history", "items"]:
            if k not in candidate:
                continue
            out = _unwrap(candidate.get(k), depth + 1)
            if out:
                return out

        for v in candidate.values():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                return _filter_dicts(v)

        return []

    return _unwrap(obj)