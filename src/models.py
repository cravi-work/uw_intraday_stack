from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import hashlib
import json

@dataclass
class Prediction:
    bias: float
    confidence: float
    prob_up: float
    prob_down: float
    prob_flat: float
    model_hash: str
    meta: Dict[str, Any]

def bounded_additive_score(
    features: Dict[str, Optional[float]],
    data_quality_score: float,
    weights: Dict[str, float],
    confidence_cap: float = 1.0,
    min_confidence: float = 0.0,
    neutral_threshold: float = 0.15,
    direction_margin: float = 0.05,
    min_flat_prob: float = 0.20,
    max_flat_prob: float = 0.80,
    flat_from_data_quality_scale: float = 1.0,
) -> Prediction:
    
    score_sum = 0.0
    total_weight = 0.0
    present_weight = 0.0
    missing_keys = []

    for key, w in weights.items():
        val = features.get(key)
        total_weight += abs(w)
        if val is not None:
            score_sum += val * w
            present_weight += abs(w)
        else:
            missing_keys.append(key)

    if total_weight > 0:
        coverage = present_weight / total_weight
    else:
        coverage = 0.0
        
    dq_eff = min(data_quality_score, coverage)

    # 1. Calculate Raw Bias
    raw_bias = score_sum / total_weight if total_weight > 0 else 0.0
    
    # 2. Mandatory Clamp
    raw_bias = max(-1.0, min(1.0, raw_bias))
    
    # 3. Confidence Logic
    if coverage < 0.4:
        confidence = 0.0
    else:
        confidence = abs(raw_bias) * dq_eff
    confidence = max(min_confidence, min(confidence, confidence_cap))

    # 4. Probability Shaping
    flat_prob = min_flat_prob + (max_flat_prob - min_flat_prob) * (1.0 - dq_eff * flat_from_data_quality_scale)
    flat_prob = min(max_flat_prob, max(min_flat_prob, flat_prob))
    remaining = 1.0 - flat_prob
    
    # 5. Direction Logic (Strict Margin Enforcement)
    # Must exceed neutral AND margin to lean
    if abs(raw_bias) < (neutral_threshold + direction_margin):
        p_up = remaining / 2.0
        p_down = remaining / 2.0
    elif raw_bias > 0:
        p_up = remaining * (0.5 + (raw_bias / 2.0))
        p_down = remaining - p_up
    else:
        p_down = remaining * (0.5 + (abs(raw_bias) / 2.0))
        p_up = remaining - p_down

    # 6. Residual Patching
    p_up = round(p_up, 4)
    p_down = round(p_down, 4)
    flat_prob = round(flat_prob, 4)
    
    total = p_up + p_down + flat_prob
    diff = 1.0 - total
    
    if abs(diff) > 1e-9:
        if flat_prob >= p_up and flat_prob >= p_down:
            flat_prob += diff
        elif p_up >= p_down:
            p_up += diff
        else:
            p_down += diff
            
    # 7. Strong Audit Hash
    # Includes all structural parameters, not just weights.
    config_state = {
        "weights": weights,
        "neutral_threshold": neutral_threshold,
        "direction_margin": direction_margin,
        "flat_params": [min_flat_prob, max_flat_prob, flat_from_data_quality_scale],
        "conf_params": [min_confidence, confidence_cap],
        "version": "phase0_frozen" 
    }
    model_hash = hashlib.sha256(json.dumps(config_state, sort_keys=True).encode()).hexdigest()[:16]

    return Prediction(
        bias=raw_bias,
        confidence=confidence,
        prob_up=round(p_up, 4),
        prob_down=round(p_down, 4),
        prob_flat=round(flat_prob, 4),
        model_hash=model_hash,
        meta={
            "coverage": round(coverage, 2),
            "dq_eff": round(dq_eff, 2),
            "missing_keys": missing_keys
        }
    )