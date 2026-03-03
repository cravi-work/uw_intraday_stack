# src/models.py
from __future__ import annotations

from typing import Dict, Any, List, Optional, Tuple, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
import hashlib
import json
import math


class SessionState(str, Enum):
    # CL-07 INTERFACE FREEZE: Do not rename or remove these values
    PREMARKET = "PREMARKET"
    RTH = "RTH"
    AFTERHOURS = "AFTERHOURS"
    CLOSED = "CLOSED"


class DataQualityState(str, Enum):
    # CL-07 INTERFACE FREEZE: Do not rename or remove these values
    VALID = "VALID"
    STALE = "STALE"
    PARTIAL = "PARTIAL"
    INVALID = "INVALID"
    DEGRADED = "DEGRADED"


class SignalState(str, Enum):
    # CL-07 INTERFACE FREEZE: Do not rename or remove these values
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"
    NO_SIGNAL = "NO_SIGNAL"


class ConfidenceState(str, Enum):
    # CL-07 INTERFACE FREEZE: Do not rename or remove these values
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    DEGRADED = "DEGRADED"
    UNKNOWN = "UNKNOWN"


class RiskGateStatus(str, Enum):
    # CL-07 INTERFACE FREEZE: Do not rename or remove these values
    PASS = "PASS"
    BLOCKED = "BLOCKED"
    DEGRADED = "DEGRADED"


class HorizonKind(str, Enum):
    # CL-07 INTERFACE FREEZE: Do not rename or remove these values
    FIXED = "FIXED"
    TO_CLOSE = "TO_CLOSE"


class OODState(str, Enum):
    IN_DISTRIBUTION = "IN_DISTRIBUTION"
    DEGRADED = "DEGRADED"
    OUT_OF_DISTRIBUTION = "OUT_OF_DISTRIBUTION"
    UNKNOWN = "UNKNOWN"


class ReplayMode(str, Enum):
    LIVE_LIKE_OBSERVED = "LIVE_LIKE_OBSERVED"
    RESEARCH_RESTATED = "RESEARCH_RESTATED"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class DecisionGate:
    # CL-07 INTERFACE FREEZE: Schema contract properties
    data_quality_state: DataQualityState
    risk_gate_status: RiskGateStatus
    decision_state: SignalState
    blocked_reasons: Tuple[str, ...] = field(default_factory=tuple)
    degraded_reasons: Tuple[str, ...] = field(default_factory=tuple)
    alignment_violations: Tuple[str, ...] = field(default_factory=tuple)
    critical_endpoints_missing: Tuple[int, ...] = field(default_factory=tuple)
    critical_features_missing: Tuple[str, ...] = field(default_factory=tuple)  # CL-05 Traceability
    validation_eligible: bool = True

    # Task 11: Structurally guarantee that a BLOCKED risk gate natively forces DataQuality to INVALID
    # unless explicitly overridden. Eliminates the "Valid but Blocked" contradiction.
    def block(self, reason: str, invalid: bool = True, missing_features: List[str] = None) -> "DecisionGate":
        return replace(
            self,
            risk_gate_status=RiskGateStatus.BLOCKED,
            decision_state=SignalState.NO_SIGNAL,
            data_quality_state=DataQualityState.INVALID if invalid else self.data_quality_state,
            blocked_reasons=self.blocked_reasons + (reason,),
            validation_eligible=False,
            critical_features_missing=tuple(missing_features) if missing_features else self.critical_features_missing,
        )

    def degrade(self, reason: str, partial: bool = False) -> "DecisionGate":
        new_status = RiskGateStatus.DEGRADED if self.risk_gate_status == RiskGateStatus.PASS else self.risk_gate_status
        return replace(
            self,
            risk_gate_status=new_status,
            data_quality_state=DataQualityState.PARTIAL if partial else self.data_quality_state,
            degraded_reasons=self.degraded_reasons + (reason,),
        )


@dataclass(frozen=True)
class PredictionTargetSpec:
    target_name: str
    target_version: str
    class_labels: Tuple[str, str, str] = ("UP", "DOWN", "FLAT")
    horizon_kind: HorizonKind = HorizonKind.FIXED
    horizon_minutes: Optional[int] = None
    flat_threshold_pct: Optional[float] = None
    probability_tolerance: float = 1e-6
    contract_source: str = "runtime_config"

    def is_valid(self) -> bool:
        labels = tuple(str(x).upper() for x in self.class_labels)
        if not self.target_name or not str(self.target_name).strip():
            return False
        if not self.target_version or not str(self.target_version).strip():
            return False
        if labels != ("UP", "DOWN", "FLAT"):
            return False
        if not math.isfinite(float(self.probability_tolerance)) or float(self.probability_tolerance) <= 0:
            return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_name": self.target_name,
            "target_version": self.target_version,
            "class_labels": list(self.class_labels),
            "horizon_kind": self.horizon_kind.value,
            "horizon_minutes": self.horizon_minutes,
            "flat_threshold_pct": self.flat_threshold_pct,
            "probability_tolerance": self.probability_tolerance,
            "contract_source": self.contract_source,
        }


@dataclass(frozen=True)
class CalibrationArtifactRef:
    artifact_name: str
    artifact_version: str
    target_name: str
    target_version: str
    bins: Tuple[float, ...]
    mapped: Tuple[float, ...]
    artifact_source: str = "runtime_config"

    def is_valid(self) -> bool:
        if not self.artifact_name or not str(self.artifact_name).strip():
            return False
        if not self.artifact_version or not str(self.artifact_version).strip():
            return False
        if not self.target_name or not str(self.target_name).strip():
            return False
        if not self.target_version or not str(self.target_version).strip():
            return False
        if len(self.bins) < 2 or len(self.bins) != len(self.mapped):
            return False

        prev_bin = None
        for b in self.bins:
            if not math.isfinite(float(b)):
                return False
            b = float(b)
            if b < 0.0 or b > 1.0:
                return False
            if prev_bin is not None and b <= prev_bin:
                return False
            prev_bin = b

        for m in self.mapped:
            if not math.isfinite(float(m)):
                return False
            m = float(m)
            if m < 0.0 or m > 1.0:
                return False
        return True

    @property
    def artifact_hash(self) -> str:
        payload = {
            "artifact_name": self.artifact_name,
            "artifact_version": self.artifact_version,
            "target_name": self.target_name,
            "target_version": self.target_version,
            "bins": list(self.bins),
            "mapped": list(self.mapped),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact_name": self.artifact_name,
            "artifact_version": self.artifact_version,
            "target_name": self.target_name,
            "target_version": self.target_version,
            "artifact_source": self.artifact_source,
            "artifact_hash": self.artifact_hash,
            "bins": list(self.bins),
            "mapped": list(self.mapped),
        }


@dataclass(frozen=True)
class ProbabilityOutput:
    raw_score: float
    raw_probability_vector: Optional[Tuple[float, float, float]]
    calibrated_probability_vector: Optional[Tuple[float, float, float]]
    confidence_state: ConfidenceState
    data_quality_state: DataQualityState
    suppression_reason: Optional[str]
    target_spec: Optional[PredictionTargetSpec]
    calibration_artifact_ref: Optional[CalibrationArtifactRef]
    ood_state: OODState

    def is_coherent(self) -> bool:
        if self.calibrated_probability_vector is None:
            return False
        p_up, p_down, p_flat = self.calibrated_probability_vector
        total = float(p_up) + float(p_down) + float(p_flat)
        return all(0.0 <= float(p) <= 1.0 and math.isfinite(float(p)) for p in (p_up, p_down, p_flat)) and abs(total - 1.0) <= (
            self.target_spec.probability_tolerance if self.target_spec is not None else 1e-6
        )

    def to_dict(self) -> Dict[str, Any]:
        def _vector_to_dict(v: Optional[Tuple[float, float, float]]) -> Optional[Dict[str, float]]:
            if v is None:
                return None
            return {"UP": float(v[0]), "DOWN": float(v[1]), "FLAT": float(v[2])}

        return {
            "raw_score": self.raw_score,
            "raw_probability_vector": _vector_to_dict(self.raw_probability_vector),
            "calibrated_probability_vector": _vector_to_dict(self.calibrated_probability_vector),
            "confidence_state": self.confidence_state.value,
            "data_quality_state": self.data_quality_state.value,
            "suppression_reason": self.suppression_reason,
            "target_spec": self.target_spec.to_dict() if self.target_spec is not None else None,
            "calibration_artifact_ref": self.calibration_artifact_ref.to_dict() if self.calibration_artifact_ref is not None else None,
            "ood_state": self.ood_state.value,
            "is_coherent": self.is_coherent(),
        }


@dataclass(frozen=True)
class Prediction:
    bias: float
    raw_score: float
    confidence: float
    confidence_state: ConfidenceState
    prob_up: Optional[float]
    prob_down: Optional[float]
    prob_flat: Optional[float]
    probability_output: ProbabilityOutput
    model_name: str
    model_version: str
    model_hash: str
    suppression_reason: Optional[str]
    ood_state: OODState
    replay_mode: ReplayMode
    meta: Dict[str, Any]
    gate: DecisionGate

    @property
    def data_quality_state(self) -> DataQualityState:
        return self.gate.data_quality_state


def _coerce_horizon_kind(value: Any) -> HorizonKind:
    if isinstance(value, HorizonKind):
        return value
    try:
        return HorizonKind(str(value).upper())
    except Exception:
        return HorizonKind.FIXED


def build_prediction_target_spec(
    model_cfg: Optional[Mapping[str, Any]],
    *,
    horizon_kind: Any = HorizonKind.FIXED,
    horizon_minutes: Optional[int] = None,
    flat_threshold_pct: Optional[float] = None,
) -> PredictionTargetSpec:
    cfg = dict(model_cfg or {})
    spec_cfg = cfg.get("target_spec") or cfg.get("target") or {}
    if not isinstance(spec_cfg, Mapping):
        spec_cfg = {}

    labels_raw = spec_cfg.get("class_labels", ("UP", "DOWN", "FLAT"))
    labels_list = list(labels_raw) if isinstance(labels_raw, Sequence) and not isinstance(labels_raw, str) else ["UP", "DOWN", "FLAT"]
    labels = tuple(str(x).upper() for x in labels_list[:3])
    if len(labels) != 3:
        labels = ("UP", "DOWN", "FLAT")

    return PredictionTargetSpec(
        target_name=str(spec_cfg.get("target_name") or spec_cfg.get("name") or "intraday_direction_3class"),
        target_version=str(spec_cfg.get("target_version") or spec_cfg.get("version") or "runtime_v1"),
        class_labels=labels,  # type: ignore[arg-type]
        horizon_kind=_coerce_horizon_kind(spec_cfg.get("horizon_kind", horizon_kind)),
        horizon_minutes=horizon_minutes,
        flat_threshold_pct=float(flat_threshold_pct) if flat_threshold_pct is not None else spec_cfg.get("flat_threshold_pct"),
        probability_tolerance=float(spec_cfg.get("probability_tolerance", 1e-6)),
        contract_source=str(spec_cfg.get("contract_source") or "runtime_config"),
    )


def build_calibration_artifact_ref(
    model_cfg: Optional[Mapping[str, Any]],
    *,
    target_spec: Optional[PredictionTargetSpec],
) -> Optional[CalibrationArtifactRef]:
    cfg = dict(model_cfg or {})
    cal_cfg = cfg.get("calibration") or {}
    if not isinstance(cal_cfg, Mapping):
        return None

    bins_raw = cal_cfg.get("bins")
    mapped_raw = cal_cfg.get("mapped")
    if not isinstance(bins_raw, Sequence) or isinstance(bins_raw, str):
        return None
    if not isinstance(mapped_raw, Sequence) or isinstance(mapped_raw, str):
        return None

    try:
        bins = tuple(float(x) for x in bins_raw)
        mapped = tuple(float(x) for x in mapped_raw)
    except (TypeError, ValueError):
        return None

    target_name = target_spec.target_name if target_spec is not None else str(cal_cfg.get("target_name") or "")
    target_version = target_spec.target_version if target_spec is not None else str(cal_cfg.get("target_version") or "")

    version = cal_cfg.get("artifact_version") or cal_cfg.get("version")
    if version is None:
        version_payload = {"bins": list(bins), "mapped": list(mapped)}
        version = hashlib.sha256(json.dumps(version_payload, sort_keys=True).encode()).hexdigest()[:12]

    return CalibrationArtifactRef(
        artifact_name=str(cal_cfg.get("artifact_name") or f"{cfg.get('model_name', 'model')}_calibration"),
        artifact_version=str(version),
        target_name=target_name,
        target_version=target_version,
        bins=bins,
        mapped=mapped,
        artifact_source=str(cal_cfg.get("artifact_source") or "runtime_config"),
    )


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


def _round_probability_vector(vector: Tuple[float, float, float], decimals: int = 4) -> Tuple[float, float, float]:
    rounded = [round(float(v), decimals) for v in vector]
    diff = round(1.0 - sum(rounded), decimals)
    if abs(diff) > 0:
        max_idx = max(range(3), key=lambda idx: rounded[idx])
        rounded[max_idx] = round(rounded[max_idx] + diff, decimals)
    return float(rounded[0]), float(rounded[1]), float(rounded[2])


def _cohere_probability_vector(
    p_up: float,
    p_down: float,
    p_flat: float,
    *,
    tolerance: float = 1e-6,
) -> Tuple[float, float, float]:
    probs = [float(p_up), float(p_down), float(p_flat)]
    if not all(math.isfinite(p) for p in probs):
        raise ValueError("Probability vector contains non-finite values")
    probs = [min(1.0, max(0.0, p)) for p in probs]
    total = sum(probs)
    if total <= 0.0:
        raise ValueError("Probability vector total must be positive")
    probs = [p / total for p in probs]
    total = sum(probs)
    if abs(total - 1.0) > tolerance:
        probs[-1] += 1.0 - total
    return _round_probability_vector((probs[0], probs[1], probs[2]))


def _confidence_state_from_value(confidence: float, gate: DecisionGate, *, suppressed: bool = False) -> ConfidenceState:
    if gate.risk_gate_status == RiskGateStatus.BLOCKED:
        return ConfidenceState.UNKNOWN
    if suppressed or gate.risk_gate_status == RiskGateStatus.DEGRADED:
        return ConfidenceState.DEGRADED
    if confidence >= 0.7:
        return ConfidenceState.HIGH
    if confidence >= 0.4:
        return ConfidenceState.MEDIUM
    return ConfidenceState.LOW


def _shape_raw_probability_vector(
    raw_bias: float,
    dq_eff: float,
    neutral_threshold: float,
    direction_margin: float,
    min_flat_prob: float,
    max_flat_prob: float,
    flat_from_data_quality_scale: float,
) -> Tuple[Tuple[float, float, float], SignalState]:
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

    return _cohere_probability_vector(p_up, p_down, flat_prob), new_signal


def _interp_piecewise_linear(x: float, bins: Sequence[float], mapped: Sequence[float]) -> float:
    x = min(1.0, max(0.0, float(x)))
    if x <= float(bins[0]):
        return float(mapped[0])
    if x >= float(bins[-1]):
        return float(mapped[-1])

    for idx in range(len(bins) - 1):
        left = float(bins[idx])
        right = float(bins[idx + 1])
        if left <= x <= right:
            left_val = float(mapped[idx])
            right_val = float(mapped[idx + 1])
            if right == left:
                return right_val
            ratio = (x - left) / (right - left)
            return left_val + ratio * (right_val - left_val)
    return float(mapped[-1])


def _calibrate_probability_vector(
    raw_vector: Tuple[float, float, float],
    raw_bias: float,
    target_spec: PredictionTargetSpec,
    calibration_artifact_ref: CalibrationArtifactRef,
    *,
    neutral_threshold: float,
    direction_margin: float,
) -> Tuple[float, float, float]:
    raw_up, raw_down, raw_flat = raw_vector
    remaining = max(0.0, 1.0 - raw_flat)
    if remaining <= target_spec.probability_tolerance:
        return _cohere_probability_vector(0.0, 0.0, 1.0, tolerance=target_spec.probability_tolerance)

    if abs(raw_bias) < (neutral_threshold + direction_margin):
        return _cohere_probability_vector(remaining / 2.0, remaining / 2.0, raw_flat, tolerance=target_spec.probability_tolerance)

    dom_raw = raw_up / remaining if raw_bias > 0 else raw_down / remaining
    dom_cal = _interp_piecewise_linear(dom_raw, calibration_artifact_ref.bins, calibration_artifact_ref.mapped)

    if raw_bias > 0:
        p_up = remaining * dom_cal
        p_down = remaining - p_up
    else:
        p_down = remaining * dom_cal
        p_up = remaining - p_down

    return _cohere_probability_vector(p_up, p_down, raw_flat, tolerance=target_spec.probability_tolerance)


def _json_safe_config_state(config_state: Dict[str, Any]) -> Dict[str, Any]:
    def _convert(value: Any) -> Any:
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, Mapping):
            return {str(k): _convert(v) for k, v in value.items()}
        if isinstance(value, tuple):
            return [_convert(v) for v in value]
        if isinstance(value, list):
            return [_convert(v) for v in value]
        if hasattr(value, "to_dict"):
            return _convert(value.to_dict())
        return value

    return _convert(config_state)


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
    *,
    model_name: str = "bounded_additive_score",
    model_version: str = "UNSPECIFIED",
    target_spec: Optional[PredictionTargetSpec] = None,
    calibration_artifact_ref: Optional[CalibrationArtifactRef] = None,
    ood_state: OODState = OODState.UNKNOWN,
    replay_mode: ReplayMode = ReplayMode.UNKNOWN,
) -> Prediction:

    safe_model_name = str(model_name or "bounded_additive_score")
    safe_model_version = str(model_version or "UNSPECIFIED")

    # Task 11: When gate is blocked, forcefully zero out confidence and set state to UNKNOWN.
    if gate.risk_gate_status == RiskGateStatus.BLOCKED:
        gate = replace(gate, decision_state=SignalState.NO_SIGNAL)
        missing = list(gate.critical_features_missing) if gate.critical_features_missing else list(weights.keys())
        probability_output = ProbabilityOutput(
            raw_score=0.0,
            raw_probability_vector=None,
            calibrated_probability_vector=None,
            confidence_state=ConfidenceState.UNKNOWN,
            data_quality_state=gate.data_quality_state,
            suppression_reason="RISK_GATE_BLOCKED",
            target_spec=target_spec,
            calibration_artifact_ref=calibration_artifact_ref,
            ood_state=ood_state,
        )
        config_state = {
            "weights": weights,
            "version": "phase0_probability_contract",
            "model_name": safe_model_name,
            "model_version": safe_model_version,
            "blocked": True,
        }
        model_hash = hashlib.sha256(json.dumps(_json_safe_config_state(config_state), sort_keys=True).encode()).hexdigest()[:16]
        meta = {
            "coverage": 0.0,
            "dq_eff": 0.0,
            "missing_keys": sorted(missing),
            "probability_contract": probability_output.to_dict(),
        }
        return Prediction(
            bias=0.0,
            raw_score=0.0,
            confidence=0.0,
            confidence_state=ConfidenceState.UNKNOWN,
            prob_up=None,
            prob_down=None,
            prob_flat=None,
            probability_output=probability_output,
            model_name=safe_model_name,
            model_version=safe_model_version,
            model_hash=model_hash,
            suppression_reason="RISK_GATE_BLOCKED",
            ood_state=ood_state,
            replay_mode=replay_mode,
            meta=meta,
            gate=gate,
        )

    score_sum = 0.0
    total_weight = 0.0
    present_weight = 0.0
    missing_keys: List[str] = []

    for key, w in weights.items():
        val = features.get(key)
        total_weight += abs(w)
        if val is not None and math.isfinite(val):
            score_sum += float(val) * float(w)
            present_weight += abs(w)
        else:
            missing_keys.append(key)

    coverage = present_weight / total_weight if total_weight > 0 else 0.0
    dq_eff = min(float(data_quality_score), coverage)

    raw_bias = score_sum / total_weight if total_weight > 0 else 0.0
    raw_bias = max(-1.0, min(1.0, raw_bias))

    confidence = 0.0 if coverage < 0.4 else abs(raw_bias) * dq_eff
    confidence = max(float(min_confidence), min(float(confidence), float(confidence_cap)))

    raw_vector, new_signal = _shape_raw_probability_vector(
        raw_bias,
        dq_eff,
        neutral_threshold,
        direction_margin,
        min_flat_prob,
        max_flat_prob,
        flat_from_data_quality_scale,
    )
    gate = replace(gate, decision_state=new_signal)

    suppression_reason: Optional[str] = None
    calibrated_vector: Optional[Tuple[float, float, float]] = None

    if ood_state == OODState.OUT_OF_DISTRIBUTION:
        suppression_reason = "OOD_REJECTION"
    elif target_spec is None:
        suppression_reason = "MISSING_TARGET_SPEC"
    elif not target_spec.is_valid():
        suppression_reason = "INVALID_TARGET_SPEC"
    elif calibration_artifact_ref is None:
        suppression_reason = "MISSING_CALIBRATION_ARTIFACT"
    elif not calibration_artifact_ref.is_valid():
        suppression_reason = "INVALID_CALIBRATION_ARTIFACT"
    elif calibration_artifact_ref.target_name != target_spec.target_name or calibration_artifact_ref.target_version != target_spec.target_version:
        suppression_reason = "CALIBRATION_TARGET_MISMATCH"
    else:
        calibrated_vector = _calibrate_probability_vector(
            raw_vector,
            raw_bias,
            target_spec,
            calibration_artifact_ref,
            neutral_threshold=neutral_threshold,
            direction_margin=direction_margin,
        )

    conf_state = _confidence_state_from_value(confidence, gate, suppressed=suppression_reason is not None)
    probability_output = ProbabilityOutput(
        raw_score=raw_bias,
        raw_probability_vector=raw_vector,
        calibrated_probability_vector=calibrated_vector,
        confidence_state=conf_state,
        data_quality_state=gate.data_quality_state,
        suppression_reason=suppression_reason,
        target_spec=target_spec,
        calibration_artifact_ref=calibration_artifact_ref,
        ood_state=ood_state,
    )

    config_state = {
        "weights": weights,
        "neutral_threshold": neutral_threshold,
        "direction_margin": direction_margin,
        "flat_params": [min_flat_prob, max_flat_prob, flat_from_data_quality_scale],
        "conf_params": [min_confidence, confidence_cap],
        "model_name": safe_model_name,
        "model_version": safe_model_version,
        "target_spec": target_spec,
        "calibration_artifact_ref": calibration_artifact_ref,
        "ood_state": ood_state,
        "replay_mode": replay_mode,
        "version": "phase0_probability_contract",
    }
    model_hash = hashlib.sha256(json.dumps(_json_safe_config_state(config_state), sort_keys=True).encode()).hexdigest()[:16]

    meta = {
        "coverage": round(coverage, 2),
        "dq_eff": round(dq_eff, 2),
        "missing_keys": sorted(missing_keys),
        "raw_probability_vector": {
            "UP": raw_vector[0],
            "DOWN": raw_vector[1],
            "FLAT": raw_vector[2],
        },
        "probability_contract": probability_output.to_dict(),
    }

    prob_up = prob_down = prob_flat = None
    if calibrated_vector is not None:
        prob_up, prob_down, prob_flat = calibrated_vector

    return Prediction(
        bias=raw_bias,
        raw_score=raw_bias,
        confidence=confidence,
        confidence_state=conf_state,
        prob_up=prob_up,
        prob_down=prob_down,
        prob_flat=prob_flat,
        probability_output=probability_output,
        model_name=safe_model_name,
        model_version=safe_model_version,
        model_hash=model_hash,
        suppression_reason=suppression_reason,
        ood_state=ood_state,
        replay_mode=replay_mode,
        meta=meta,
        gate=gate,
    )


KNOWN_FEATURE_KEYS = frozenset(
    {
        "spot",
        "smart_whale_pressure",
        "dealer_vanna",
        "dealer_charm",
        "net_gex_sign",
        "net_gamma_exposure_notional",
        "oi_pressure",
        "darkpool_pressure",
        "litflow_pressure",
        "vol_term_slope",
        "vol_skew",
        "iv_rank",
    }
)
