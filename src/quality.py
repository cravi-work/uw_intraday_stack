"""
src/quality.py
Robust Data Quality computation with Age Decay and Critical Path enforcement.
"""
from typing import Dict, List, Any, Optional

def compute_weighted_data_quality(
    lineage_map: Dict[str, Dict[str, Any]], # {key: {"state": "FRESH", "age_sec": 120}}
    critical_keys: List[str],
    major_keys: List[str],
    max_stale_age_sec: int = 5400 # 90 mins before STALE becomes worthless
) -> float:
    
    if not lineage_map:
        return 0.0
        
    # 1. Critical Path Kill Switch
    for k in critical_keys:
        info = lineage_map.get(k, {})
        if info.get("state") in ("MISSING", "INVALID"):
            return 0.0 
            
    # 2. Major Component Cap
    major_broken = False
    for k in major_keys:
        info = lineage_map.get(k, {})
        if info.get("state") in ("MISSING", "INVALID"):
            major_broken = True
            break
            
    # 3. Age-Weighted Scoring
    total_score = 0.0
    count = 0
    
    BASE_SCORES = {
        "FRESH": 1.0,
        "STALE": 0.5,
        "FALLBACK": 0.25,
        "MISSING": 0.0,
        "INVALID": 0.0
    }
    
    for info in lineage_map.values():
        state = info.get("state", "MISSING")
        age = info.get("age_sec")
        
        base = BASE_SCORES.get(state, 0.0)
        
        if state == "STALE" and age is not None:
            decay_factor = max(0.0, 1.0 - (age / max_stale_age_sec))
            score = base * decay_factor
        else:
            score = base
            
        total_score += score
        count += 1
        
    avg_score = total_score / count if count > 0 else 0.0
    
    if major_broken:
        return min(0.5, round(avg_score, 2))
        
    return round(avg_score, 2)