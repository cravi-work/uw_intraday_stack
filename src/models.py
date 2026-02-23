from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field, replace
from enum import Enum
import hashlib
import json
import math

class SessionState(str, Enum):
    PREMARKET = "PREMARKET"
    RTH = "RTH"
    AFTERHOURS = "AFTERHOURS"
    CLOSED = "CLOSED"

class DataQualityState(str, Enum):
    VALID = "VALID"
    STALE = "STALE"
    PARTIAL = "PARTIAL"
    INVALID = "INVALID"
    DEGRADED = "DEGRADED"

class SignalState(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"
    NO_SIGNAL = "NO_SIGNAL"

class ConfidenceState(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    DEGRADED = "DEGRADED"
    UNKNOWN = "UNKNOWN"

class RiskGateStatus(str, Enum):
    PASS = "PASS"
    BLOCKED = "BLOCKED"
    DEGRADED = "DEGRADED"

class HorizonKind(str, Enum):
    FIXED = "FIXED"
    TO_CLOSE = "TO_CLOSE"

@dataclass(frozen=True)
class DecisionGate:
    data_quality_state: DataQualityState
    risk_gate_status: RiskGateStatus
    decision_state: SignalState
    blocked_reasons: Tuple[str, ...] = field(default_factory=tuple)
    degraded_reasons: Tuple[str, ...] = field(default_factory=tuple)
    alignment_violations: Tuple[str, ...] = field(default_factory=tuple)
    critical_endpoints_missing: Tuple[int, ...] = field(default_factory=tuple)
    validation_eligible: bool = True

    def block(self, reason: str, invalid: bool = False) -> 'DecisionGate':
        return replace(
            self,
            risk_gate_status=RiskGateStatus.BLOCKED,
            decision_state=SignalState.NO_SIGNAL,
            data_quality_state=DataQualityState.INVALID if invalid else self.data_quality_state,
            blocked_reasons=self.blocked_reasons + (reason,),
            validation_eligible=False
        )

    def degrade(self, reason: str, partial: bool = False) -> 'DecisionGate':
        new_status = RiskGateStatus.DEGRADED if self.risk_gate_status == RiskGateStatus.PASS else self.risk_gate_status
        return replace(
            self,
            risk_gate_status=new_status,
            data_quality_state=DataQualityState.PARTIAL if partial else self.data_quality_state,
            degraded_reasons=self.degraded_reasons + (reason,)
        )

@dataclass(frozen=True)
class Prediction:
    bias: float
    confidence: float
    confidence_state: ConfidenceState
    prob_up: Optional[float]
    prob_down: Optional[float]
    prob_flat: Optional[float]
    model_name: str
    model_version: str
    model_hash: str
    meta: Dict[str, Any]
    gate: DecisionGate

def predicted_class(prob_up: Optional[float], prob_down: Optional[float], prob_flat: Optional[float]) -> str:
    if prob_up is None or prob_down is None or prob_flat is None:
        return "FLAT"
    try:
        if math.isnan(prob_up) or math.isnan(prob_down) or math.isnan(prob_flat):
            return "FLAT"
    except TypeError:
        return "FLAT"

    if prob_flat >= prob_up and prob_flat >= prob_down: 
        return "FLAT"
    if prob_up >= prob_down: 
        return "UP"
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
    
    if gate.risk_gate_status == RiskGateStatus.BLOCKED:
        gate = replace(gate, decision_state=SignalState.NO_SIGNAL)
        return Prediction(
            bias=0.0, 
            confidence=0.0, 
            confidence_state=ConfidenceState.UNKNOWN,
            prob_up=None, 
            prob_down=None, 
            prob_flat=None,
            model_name="phase0_additive", 
            model_version="1.0", 
            model_hash="BLOCKED",
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
        if val is not None and math.isfinite(val):
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
        new_signal = SignalState.NEUTRAL
    elif raw_bias > 0:
        p_up = remaining * (0.5 + (raw_bias / 2.0))
        p_down = remaining - p_up
        new_signal = SignalState.LONG
    else:
        p_down = remaining * (0.5 + (abs(raw_bias) / 2.0))
        p_up = remaining - p_down
        new_signal = SignalState.SHORT

    gate = replace(gate, decision_state=new_signal if gate.risk_gate_status != RiskGateStatus.BLOCKED else gate.decision_state)

    if confidence >= 0.7:
        conf_state = ConfidenceState.HIGH
    elif confidence >= 0.4:
        conf_state = ConfidenceState.MEDIUM
    else:
        conf_state = ConfidenceState.LOW

    if gate.risk_gate_status == RiskGateStatus.DEGRADED:
        conf_state = ConfidenceState.DEGRADED

    p_up, p_down, flat_prob = round(p_up, 4), round(p_down, 4), round(flat_prob, 4)
    
    diff = 1.0 - (p_up + p_down + flat_prob)
    if abs(diff) > 1e-9:
        if flat_prob >= p_up and flat_prob >= p_down: 
            flat_prob += diff
        elif p_up >= p_down: 
            p_up += diff
        else: 
            p_down += diff
            
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
        confidence_state=conf_state,
        prob_up=round(p_up, 4),
        prob_down=round(p_down, 4), 
        prob_flat=round(flat_prob, 4),
        model_name="phase0_additive", 
        model_version="1.0", 
        model_hash=model_hash,
        meta={"coverage": round(coverage, 2), "dq_eff": round(dq_eff, 2), "missing_keys": missing_keys},
        gate=gate
    )