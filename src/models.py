from typing import Dict, Any, List, Optional, Literal
from dataclasses import dataclass, field
import hashlib
import json
import math

PredictionLabel = Literal["UP", "DOWN", "FLAT"]
HorizonKind = Literal["FIXED", "TO_CLOSE"]
SessionState = Literal["PREMARKET", "RTH", "AFTERHOURS", "CLOSED"]
DataQualityState = Literal["VALID", "STALE", "PARTIAL", "INVALID"]
DecisionState = Literal["LONG", "SHORT", "NEUTRAL", "NO_TRADE"]
RiskGateStatus = Literal["PASS", "BLOCKED", "DEGRADED"]

@dataclass
class DecisionGate:
    data_quality_state: DataQualityState
    risk_gate_status: RiskGateStatus
    decision_state: DecisionState
    blocked_reasons: List[str] = field(default_factory=list)
    degraded_reasons: List[str] = field(default_factory=list)
    validation_eligible: bool = True

@dataclass
class Prediction:
    bias: float
    confidence: float
    prob_up: float
    prob_down: float
    prob_flat: float
    model_name: str
    model_version: str
    model_hash: str
    meta: Dict[str, Any]
    gate: DecisionGate

def predicted_class(prob_up: float, prob_down: float, prob_flat: float) -> PredictionLabel:
    """
    Determines the discrete predicted label from continuous probabilities.
    Exact Tie Rule: FLAT > UP > DOWN.
    Gracefully falls back to FLAT if probabilities are invalid (NaN/None).
    """
    try:
        if math.isnan(prob_up) or math.isnan(prob_down) or math.isnan(prob_flat):
            return "FLAT"
    except TypeError:
        return "FLAT"

    # Deterministic resolution order for ties
    if prob_flat >= prob_up and prob_flat >= prob_down: return "FLAT"
    if prob_up >= prob_down: return "UP"
    return "DOWN"

def bounded_additive_score(
    features: Dict[str, Optional[float]],
    data_quality_score: float,
    weights: Dict[str, float],
    gate: DecisionGate,
    confidence_cap: float = 1.0,
    min_confidence: float = 0.0,
    neutral_threshold: float = 0.15,
    direction_margin: float = 0.05,
    min_flat_prob: float = 0.20,
    max_flat_prob: float = 0.80,
    flat_from_data_quality_scale: float = 1.0,
) -> Prediction:
    
    # 1. Critical Risk Block Path
    if gate.risk_gate_status == "BLOCKED":
        gate.decision_state = "NO_TRADE"
        return Prediction(
            bias=0.0, confidence=0.0, prob_up=0.0, prob_down=0.0, prob_flat=1.0,
            model_name="phase0_additive", model_version="1.0", model_hash="BLOCKED",
            meta={"coverage": 0.0, "dq_eff": 0.0, "missing_keys": list(weights.keys())},
            gate=gate
        )

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

    coverage = present_weight / total_weight if total_weight > 0 else 0.0
    dq_eff = min(data_quality_score, coverage)

    raw_bias = score_sum / total_weight if total_weight > 0 else 0.0
    raw_bias = max(-1.0, min(1.0, raw_bias))
    
    confidence = 0.0 if coverage < 0.4 else abs(raw_bias) * dq_eff
    confidence = max(min_confidence, min(confidence, confidence_cap))

    flat_prob = min_flat_prob + (max_flat_prob - min_flat_prob) * (1.0 - dq_eff * flat_from_data_quality_scale)
    flat_prob = min(max_flat_prob, max(min_flat_prob, flat_prob))
    remaining = 1.0 - flat_prob
    
    if abs(raw_bias) < (neutral_threshold + direction_margin):
        p_up = remaining / 2.0
        p_down = remaining / 2.0
        if gate.risk_gate_status != "BLOCKED": gate.decision_state = "NEUTRAL"
    elif raw_bias > 0:
        p_up = remaining * (0.5 + (raw_bias / 2.0))
        p_down = remaining - p_up
        if gate.risk_gate_status != "BLOCKED": gate.decision_state = "LONG"
    else:
        p_down = remaining * (0.5 + (abs(raw_bias) / 2.0))
        p_up = remaining - p_down
        if gate.risk_gate_status != "BLOCKED": gate.decision_state = "SHORT"

    p_up, p_down, flat_prob = round(p_up, 4), round(p_down, 4), round(flat_prob, 4)
    
    diff = 1.0 - (p_up + p_down + flat_prob)
    if abs(diff) > 1e-9:
        if flat_prob >= p_up and flat_prob >= p_down: flat_prob += diff
        elif p_up >= p_down: p_up += diff
        else: p_down += diff
            
    config_state = {
        "weights": weights, "neutral_threshold": neutral_threshold,
        "direction_margin": direction_margin,
        "flat_params": [min_flat_prob, max_flat_prob, flat_from_data_quality_scale],
        "conf_params": [min_confidence, confidence_cap],
        "version": "phase0_frozen" 
    }
    model_hash = hashlib.sha256(json.dumps(config_state, sort_keys=True).encode()).hexdigest()[:16]

    return Prediction(
        bias=raw_bias, confidence=confidence, prob_up=round(p_up, 4),
        prob_down=round(p_down, 4), prob_flat=round(flat_prob, 4),
        model_name="phase0_additive", model_version="1.0", model_hash=model_hash,
        meta={"coverage": round(coverage, 2), "dq_eff": round(dq_eff, 2), "missing_keys": missing_keys},
        gate=gate
    )