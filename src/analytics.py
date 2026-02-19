from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# FIXED: Changed _find_list_of_dicts to _grab_list to match features.py
from .features import _as_float, _find_first, _grab_list


@dataclass(frozen=True)
class Level:
    level_type: str
    price: Optional[float]
    magnitude: Optional[float]
    meta: Dict[str, Any]


def volume_profile_levels(stock_volume_price_levels_payload: Any, top_n: int) -> List[Level]:
    """Volume profile shelves: HVNs + POC."""
    # FIXED: Use _grab_list
    rows = _grab_list(stock_volume_price_levels_payload)
    parsed: List[Tuple[float, float, Dict[str, Any]]] = []
    for r in rows:
        price = _as_float(_find_first(r, ["price", "p", "level"]))
        vol = _as_float(_find_first(r, ["volume", "v", "shares"]))
        if price is None or vol is None:
            continue
        parsed.append((price, vol, r))
    parsed.sort(key=lambda x: x[1], reverse=True)
    if not parsed:
        return []
    out: List[Level] = []
    poc = parsed[0]
    out.append(Level("VP_POC", poc[0], poc[1], {"raw": poc[2]}))
    for price, vol, raw in parsed[:top_n]:
        out.append(Level("VP_NODE", price, vol, {"raw": raw}))
    return out


def darkpool_magnets(darkpool_payload: Any, band_width: float, top_n: int) -> List[Level]:
    """Darkpool magnets via price band aggregation."""
    # FIXED: Use _grab_list
    rows = _grab_list(darkpool_payload)
    bands: Dict[float, float] = {}
    samples: Dict[float, List[Dict[str, Any]]] = {}
    for r in rows:
        price = _as_float(_find_first(r, ["price", "p"]))
        prem = _as_float(_find_first(r, ["premium", "notional", "value", "dollar_value"]))
        if price is None or prem is None:
            continue
        center = round(price / band_width) * band_width
        bands[center] = bands.get(center, 0.0) + prem
        samples.setdefault(center, []).append(r)

    ranked = sorted(bands.items(), key=lambda kv: abs(kv[1]), reverse=True)
    out: List[Level] = []
    for center, mag in ranked[:top_n]:
        out.append(Level("DARK_POOL_MAGNET", float(center), float(mag), {"band_width": band_width, "n": len(samples.get(center, []))}))
    return out


def litflow_shelves(lit_payload: Any, band_width: float, top_n: int) -> List[Level]:
    """Lit-flow shelves via price band aggregation."""
    # FIXED: Use _grab_list
    rows = _grab_list(lit_payload)
    bands: Dict[float, float] = {}
    for r in rows:
        price = _as_float(_find_first(r, ["price", "p"]))
        prem = _as_float(_find_first(r, ["premium", "notional", "value", "dollar_value"]))
        if price is None or prem is None:
            continue
        center = round(price / band_width) * band_width
        bands[center] = bands.get(center, 0.0) + prem
    ranked = sorted(bands.items(), key=lambda kv: abs(kv[1]), reverse=True)
    return [Level("LIT_SHELF", float(center), float(mag), {"band_width": band_width}) for center, mag in ranked[:top_n]]


def oi_walls(oi_per_strike_payload: Any, top_n: int) -> List[Level]:
    """Call wall / Put wall from top OI strikes."""
    # FIXED: Use _grab_list
    rows = _grab_list(oi_per_strike_payload)
    calls: List[Tuple[float, float, Dict[str, Any]]] = []
    puts: List[Tuple[float, float, Dict[str, Any]]] = []
    for r in rows:
        strike = _as_float(_find_first(r, ["strike", "k", "Strike"]))
        call_oi = _as_float(_find_first(r, ["call_oi", "calls_oi", "callOI", "call_open_interest"]))
        put_oi = _as_float(_find_first(r, ["put_oi", "puts_oi", "putOI", "put_open_interest"]))
        if strike is None:
            continue
        if call_oi is not None:
            calls.append((strike, call_oi, r))
        if put_oi is not None:
            puts.append((strike, put_oi, r))

    out: List[Level] = []
    calls.sort(key=lambda x: x[1], reverse=True)
    puts.sort(key=lambda x: x[1], reverse=True)
    for strike, oi, raw in calls[:top_n]:
        out.append(Level("CALL_OI", float(strike), float(oi), {"raw": raw}))
    for strike, oi, raw in puts[:top_n]:
        out.append(Level("PUT_OI", float(strike), float(oi), {"raw": raw}))

    if calls:
        cw = calls[0]
        out.append(Level("CALL_WALL", float(cw[0]), float(cw[1]), {"raw": cw[2]}))
    if puts:
        pw = puts[0]
        out.append(Level("PUT_WALL", float(pw[0]), float(pw[1]), {"raw": pw[2]}))
    return out


def gex_concentrations(spot_exposures_strike_payload: Any, top_n: int) -> List[Level]:
    """Per-strike gamma exposure concentrations."""
    # FIXED: Use _grab_list
    rows = _grab_list(spot_exposures_strike_payload)
    parsed: List[Tuple[float, float, Dict[str, Any]]] = []
    for r in rows:
        strike = _as_float(_find_first(r, ["strike", "k", "Strike"]))
        gex = _as_float(_find_first(r, ["gex", "gamma_exposure", "gammaExposure", "exposure", "value"]))
        if strike is None or gex is None:
            continue
        parsed.append((strike, gex, r))
    parsed.sort(key=lambda x: abs(x[1]), reverse=True)
    out = [Level("GEX_STRIKE", float(s), float(g), {"raw": raw}) for s, g, raw in parsed[:top_n]]
    return out


def gex_flip_estimate(spot_exposures_strike_payload: Any) -> Optional[Level]:
    """Gamma flip estimate via cumulative exposure sign change across strikes."""
    # FIXED: Use _grab_list
    rows = _grab_list(spot_exposures_strike_payload)
    parsed: List[Tuple[float, float]] = []
    for r in rows:
        strike = _as_float(_find_first(r, ["strike", "k", "Strike"]))
        gex = _as_float(_find_first(r, ["gex", "gamma_exposure", "gammaExposure", "exposure", "value"]))
        if strike is None or gex is None:
            continue
        parsed.append((strike, gex))
    if len(parsed) < 3:
        return None
    parsed.sort(key=lambda x: x[0])

    cum = 0.0
    prev_s = None
    prev_c = None
    flip = None
    for s, g in parsed:
        cum += g
        if prev_c is not None and ((prev_c <= 0 < cum) or (prev_c >= 0 > cum)):
            if cum == prev_c:
                flip = s
            else:
                t = (0 - prev_c) / (cum - prev_c)
                flip = prev_s + t * (s - prev_s)
            break
        prev_s, prev_c = s, cum
    if flip is None:
        return None
    return Level("GEX_FLIP", float(flip), 0.0, {"method": "cumulative_cross"})


def max_pain_level(max_pain_payload: Any) -> Optional[Level]:
    """Max pain level."""
    if isinstance(max_pain_payload, dict):
        v = _as_float(_find_first(max_pain_payload, ["max_pain", "maxPain", "price"]))
        if v is not None:
            return Level("MAX_PAIN", float(v), None, {"schema": "dict"})
    # FIXED: Use _grab_list
    rows = _grab_list(max_pain_payload)
    if rows:
        v = _as_float(_find_first(rows[0], ["max_pain", "maxPain", "price"]))
        if v is not None:
            return Level("MAX_PAIN", float(v), None, {"schema": "list[0]"})
    return None