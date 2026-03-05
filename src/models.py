# src/models.py
from __future__ import annotations

from typing import Dict, Any, List, Optional, Tuple, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
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
class LabelContractSpec:
    label_version: str
    session_boundary_rule: str
    flat_threshold_pct: float
    flat_threshold_policy: str = "ABS_RETURN_BAND"
    threshold_policy_version: str = "runtime_v1"
    neutral_threshold: Optional[float] = None
    direction_margin: Optional[float] = None
    contract_source: str = "runtime_config"

    def is_valid(self) -> bool:
        if not self.label_version or not str(self.label_version).strip():
            return False
        if not self.session_boundary_rule or not str(self.session_boundary_rule).strip():
            return False
        if self.session_boundary_rule not in {"TRUNCATE_TO_SESSION_CLOSE", "REQUIRE_TARGET_WITHIN_SESSION"}:
            return False
        if not math.isfinite(float(self.flat_threshold_pct)) or float(self.flat_threshold_pct) < 0.0:
            return False
        if not self.flat_threshold_policy or not str(self.flat_threshold_policy).strip():
            return False
        if not self.threshold_policy_version or not str(self.threshold_policy_version).strip():
            return False
        if self.neutral_threshold is not None and not math.isfinite(float(self.neutral_threshold)):
            return False
        if self.direction_margin is not None and not math.isfinite(float(self.direction_margin)):
            return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label_version": self.label_version,
            "session_boundary_rule": self.session_boundary_rule,
            "flat_threshold_pct": float(self.flat_threshold_pct),
            "flat_threshold_policy": self.flat_threshold_policy,
            "threshold_policy_version": self.threshold_policy_version,
            "neutral_threshold": self.neutral_threshold,
            "direction_margin": self.direction_margin,
            "contract_source": self.contract_source,
        }


UTC = timezone.utc
DEFAULT_CALIBRATION_PROVENANCE_FIELDS: Tuple[str, ...] = (
    "trained_from_utc",
    "trained_to_utc",
    "valid_from_utc",
    "valid_to_utc",
    "evidence_ref",
    "fit_sample_count",
)


def _normalize_optional_utc_iso(value: Any) -> Optional[str]:
    if value in (None, ""):
        return None
    try:
        if isinstance(value, datetime):
            dt_val = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
            return dt_val.astimezone(UTC).isoformat()
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return datetime.fromtimestamp(float(value), UTC).isoformat()
        raw = str(value).strip()
        if not raw:
            return None
        dt_val = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if dt_val.tzinfo is None:
            dt_val = dt_val.replace(tzinfo=UTC)
        else:
            dt_val = dt_val.astimezone(UTC)
        return dt_val.isoformat()
    except (OverflowError, OSError, TypeError, ValueError):
        return None


def _coerce_optional_positive_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    coerced = int(value)
    return coerced


@dataclass(frozen=True)
class CalibrationArtifactRef:
    artifact_name: str
    artifact_version: str
    target_name: str
    target_version: str
    bins: Tuple[float, ...]
    mapped: Tuple[float, ...]
    artifact_source: str = "runtime_config"
    scope_horizon_kind: str = "ANY"
    scope_horizon_minutes: Optional[int] = None
    scope_session: str = "ANY"
    scope_regime: str = "ANY"
    scope_replay_mode: str = "ANY"
    scope_contract_version: str = "calibration_scope/v1"
    provenance_contract_version: str = "calibration_provenance/v1"
    trained_from_utc: Optional[str] = None
    trained_to_utc: Optional[str] = None
    valid_from_utc: Optional[str] = None
    valid_to_utc: Optional[str] = None
    evidence_ref: Optional[str] = None
    fit_sample_count: Optional[int] = None

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

        scope_horizon_kind = str(self.scope_horizon_kind or "ANY").upper()
        if scope_horizon_kind not in {"ANY", HorizonKind.FIXED.value, HorizonKind.TO_CLOSE.value}:
            return False
        if scope_horizon_kind == HorizonKind.FIXED.value and self.scope_horizon_minutes is not None:
            try:
                if int(self.scope_horizon_minutes) <= 0:
                    return False
            except (TypeError, ValueError):
                return False
        if scope_horizon_kind == HorizonKind.TO_CLOSE.value and self.scope_horizon_minutes not in (None, 0):
            return False

        scope_session = str(self.scope_session or "ANY").upper()
        if scope_session not in {"ANY", SessionState.PREMARKET.value, SessionState.RTH.value, SessionState.AFTERHOURS.value, SessionState.CLOSED.value}:
            return False

        scope_replay_mode = str(self.scope_replay_mode or "ANY").upper()
        if scope_replay_mode not in {"ANY", ReplayMode.UNKNOWN.value, ReplayMode.LIVE_LIKE_OBSERVED.value, ReplayMode.RESEARCH_RESTATED.value}:
            return False

        if not str(self.scope_regime or "ANY").strip():
            return False
        if not str(self.scope_contract_version or "").strip():
            return False
        if not str(self.provenance_contract_version or "").strip():
            return False

        trained_from = _normalize_optional_utc_iso(self.trained_from_utc)
        trained_to = _normalize_optional_utc_iso(self.trained_to_utc)
        valid_from = _normalize_optional_utc_iso(self.valid_from_utc)
        valid_to = _normalize_optional_utc_iso(self.valid_to_utc)

        if self.trained_from_utc not in (None, "") and trained_from is None:
            return False
        if self.trained_to_utc not in (None, "") and trained_to is None:
            return False
        if self.valid_from_utc not in (None, "") and valid_from is None:
            return False
        if self.valid_to_utc not in (None, "") and valid_to is None:
            return False
        if trained_from is not None and trained_to is not None and trained_from > trained_to:
            return False
        if valid_from is not None and valid_to is not None and valid_from > valid_to:
            return False

        if self.evidence_ref not in (None, "") and not str(self.evidence_ref).strip():
            return False

        if self.fit_sample_count is not None:
            try:
                if int(self.fit_sample_count) <= 0:
                    return False
            except (TypeError, ValueError):
                return False
        return True

    @property
    def calibration_scope(self) -> Dict[str, Any]:
        return {
            "horizon_kind": str(self.scope_horizon_kind or "ANY").upper(),
            "horizon_minutes": None if self.scope_horizon_minutes is None else int(self.scope_horizon_minutes),
            "session": str(self.scope_session or "ANY").upper(),
            "regime": str(self.scope_regime or "ANY").upper(),
            "replay_mode": str(self.scope_replay_mode or "ANY").upper(),
            "scope_contract_version": self.scope_contract_version,
        }

    @property
    def artifact_provenance(self) -> Dict[str, Any]:
        return {
            "trained_from_utc": _normalize_optional_utc_iso(self.trained_from_utc),
            "trained_to_utc": _normalize_optional_utc_iso(self.trained_to_utc),
            "valid_from_utc": _normalize_optional_utc_iso(self.valid_from_utc),
            "valid_to_utc": _normalize_optional_utc_iso(self.valid_to_utc),
            "evidence_ref": str(self.evidence_ref).strip() if self.evidence_ref not in (None, "") else None,
            "fit_sample_count": None if self.fit_sample_count is None else int(self.fit_sample_count),
            "provenance_contract_version": self.provenance_contract_version,
        }

    def missing_provenance_fields(self, required_fields: Optional[Sequence[str]] = None) -> Tuple[str, ...]:
        required = tuple(required_fields or DEFAULT_CALIBRATION_PROVENANCE_FIELDS)
        provenance = self.artifact_provenance
        missing: list[str] = []
        for field_name in required:
            if field_name == "artifact_hash":
                continue
            if provenance.get(field_name) in (None, ""):
                missing.append(str(field_name))
        return tuple(missing)

    @property
    def artifact_hash(self) -> str:
        payload = {
            "artifact_name": self.artifact_name,
            "artifact_version": self.artifact_version,
            "target_name": self.target_name,
            "target_version": self.target_version,
            "calibration_scope": self.calibration_scope,
            "artifact_provenance": self.artifact_provenance,
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
            "calibration_scope": self.calibration_scope,
            "artifact_provenance": self.artifact_provenance,
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
    ood_reason: Optional[str] = None
    ood_policy_action: str = "emit"

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
            "ood_reason": self.ood_reason,
            "ood_policy_action": self.ood_policy_action,
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


def build_label_contract_spec(
    model_cfg: Optional[Mapping[str, Any]],
    validation_cfg: Optional[Mapping[str, Any]] = None,
    *,
    flat_threshold_pct: Optional[float] = None,
    session_boundary_rule: Optional[str] = None,
) -> LabelContractSpec:
    cfg = dict(model_cfg or {})
    val_cfg = dict(validation_cfg or {})
    label_cfg = val_cfg.get("label_contract") or cfg.get("label_contract") or {}
    if not isinstance(label_cfg, Mapping):
        label_cfg = {}

    def _maybe_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        return float(value)

    resolved_flat = flat_threshold_pct
    if resolved_flat is None:
        resolved_flat = label_cfg.get("flat_threshold_pct", val_cfg.get("flat_threshold_pct", 0.0))

    neutral_threshold = label_cfg.get("neutral_threshold", cfg.get("neutral_threshold"))
    direction_margin = label_cfg.get("direction_margin", cfg.get("direction_margin"))

    return LabelContractSpec(
        label_version=str(label_cfg.get("label_version") or val_cfg.get("label_version") or "runtime_v1"),
        session_boundary_rule=str(session_boundary_rule or label_cfg.get("session_boundary_rule") or "TRUNCATE_TO_SESSION_CLOSE"),
        flat_threshold_pct=float(resolved_flat),
        flat_threshold_policy=str(label_cfg.get("flat_threshold_policy") or "ABS_RETURN_BAND"),
        threshold_policy_version=str(label_cfg.get("threshold_policy_version") or label_cfg.get("policy_version") or "runtime_v1"),
        neutral_threshold=_maybe_float(neutral_threshold),
        direction_margin=_maybe_float(direction_margin),
        contract_source=str(label_cfg.get("contract_source") or "runtime_config"),
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

    scope_cfg = cal_cfg.get("scope") if isinstance(cal_cfg.get("scope"), Mapping) else {}
    provenance_cfg = cal_cfg.get("provenance") if isinstance(cal_cfg.get("provenance"), Mapping) else {}
    scope_horizon_minutes = scope_cfg.get("horizon_minutes")
    if scope_horizon_minutes not in (None, ""):
        try:
            scope_horizon_minutes = int(scope_horizon_minutes)
        except (TypeError, ValueError):
            return None

    fit_sample_count = provenance_cfg.get("fit_sample_count", cal_cfg.get("fit_sample_count"))
    if fit_sample_count not in (None, ""):
        try:
            fit_sample_count = int(fit_sample_count)
        except (TypeError, ValueError):
            return None

    return CalibrationArtifactRef(
        artifact_name=str(cal_cfg.get("artifact_name") or f"{cfg.get('model_name', 'model')}_calibration"),
        artifact_version=str(version),
        target_name=target_name,
        target_version=target_version,
        bins=bins,
        mapped=mapped,
        artifact_source=str(cal_cfg.get("artifact_source") or "runtime_config"),
        scope_horizon_kind=str(scope_cfg.get("horizon_kind") or "ANY").upper(),
        scope_horizon_minutes=scope_horizon_minutes,
        scope_session=str(scope_cfg.get("session") or "ANY").upper(),
        scope_regime=str(scope_cfg.get("regime") or "ANY").upper(),
        scope_replay_mode=str(scope_cfg.get("replay_mode") or "ANY").upper(),
        scope_contract_version=str(scope_cfg.get("scope_contract_version") or "calibration_scope/v1"),
        provenance_contract_version=str(provenance_cfg.get("provenance_contract_version") or cal_cfg.get("provenance_contract_version") or "calibration_provenance/v1"),
        trained_from_utc=_normalize_optional_utc_iso(provenance_cfg.get("trained_from_utc", cal_cfg.get("trained_from_utc"))),
        trained_to_utc=_normalize_optional_utc_iso(provenance_cfg.get("trained_to_utc", cal_cfg.get("trained_to_utc"))),
        valid_from_utc=_normalize_optional_utc_iso(provenance_cfg.get("valid_from_utc", cal_cfg.get("valid_from_utc"))),
        valid_to_utc=_normalize_optional_utc_iso(provenance_cfg.get("valid_to_utc", cal_cfg.get("valid_to_utc"))),
        evidence_ref=str(provenance_cfg.get("evidence_ref", cal_cfg.get("evidence_ref")) or "").strip() or None,
        fit_sample_count=fit_sample_count,
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


_DEFAULT_OOD_EMISSION_POLICY: Dict[str, Any] = {
    "degraded_confidence_scale": 0.5,
    "unknown_confidence_scale": 0.0,
    "out_confidence_scale": 0.0,
    "degraded_emit_calibrated": True,
    "unknown_emit_calibrated": False,
    "out_emit_calibrated": False,
}


def _resolve_ood_emission_policy(policy: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    resolved = dict(_DEFAULT_OOD_EMISSION_POLICY)
    if not isinstance(policy, Mapping):
        return resolved
    for key, fallback in _DEFAULT_OOD_EMISSION_POLICY.items():
        value = policy.get(key, fallback)
        if isinstance(fallback, bool):
            resolved[key] = bool(value)
        else:
            try:
                resolved[key] = float(value)
            except (TypeError, ValueError):
                resolved[key] = fallback
    return resolved


def _resolve_ood_probability_effect(
    ood_state: OODState,
    *,
    policy: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    resolved_policy = _resolve_ood_emission_policy(policy)

    if ood_state == OODState.OUT_OF_DISTRIBUTION:
        return {
            "confidence_scale": float(resolved_policy["out_confidence_scale"]),
            "suppress_calibrated_probability": not bool(resolved_policy["out_emit_calibrated"]),
            "suppression_reason": "OOD_REJECTION",
            "policy_action": "suppress",
        }

    if ood_state == OODState.UNKNOWN:
        return {
            "confidence_scale": float(resolved_policy["unknown_confidence_scale"]),
            "suppress_calibrated_probability": not bool(resolved_policy["unknown_emit_calibrated"]),
            "suppression_reason": "OOD_UNKNOWN",
            "policy_action": "suppress",
        }

    if ood_state == OODState.DEGRADED:
        return {
            "confidence_scale": float(resolved_policy["degraded_confidence_scale"]),
            "suppress_calibrated_probability": not bool(resolved_policy["degraded_emit_calibrated"]),
            "suppression_reason": "OOD_DEGRADED" if not bool(resolved_policy["degraded_emit_calibrated"]) else None,
            "policy_action": "degrade_confidence",
        }

    return {
        "confidence_scale": 1.0,
        "suppress_calibrated_probability": False,
        "suppression_reason": None,
        "policy_action": "emit",
    }


def _confidence_state_from_value(
    confidence: float,
    gate: DecisionGate,
    *,
    suppressed: bool = False,
    ood_state: OODState = OODState.IN_DISTRIBUTION,
) -> ConfidenceState:
    if gate.risk_gate_status == RiskGateStatus.BLOCKED:
        return ConfidenceState.UNKNOWN
    if ood_state == OODState.UNKNOWN:
        return ConfidenceState.UNKNOWN
    if suppressed or gate.risk_gate_status == RiskGateStatus.DEGRADED or ood_state in (OODState.DEGRADED, OODState.OUT_OF_DISTRIBUTION):
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
    ood_reason: Optional[str] = None,
    ood_policy: Optional[Mapping[str, Any]] = None,
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
            ood_reason=ood_reason,
            ood_policy_action="blocked",
        )
        config_state = {
            "weights": weights,
            "version": "phase0_probability_contract",
            "model_name": safe_model_name,
            "model_version": safe_model_version,
            "ood_policy": dict(ood_policy or {}),
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

    ood_effect = _resolve_ood_probability_effect(ood_state, policy=ood_policy)
    confidence = max(0.0, min(1.0, float(confidence) * float(ood_effect["confidence_scale"])))

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

    if ood_effect["suppress_calibrated_probability"]:
        suppression_reason = str(ood_effect["suppression_reason"])
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

    conf_state = _confidence_state_from_value(
        confidence,
        gate,
        suppressed=suppression_reason is not None,
        ood_state=ood_state,
    )
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
        ood_reason=ood_reason,
        ood_policy_action=str(ood_effect["policy_action"]),
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
        "ood_policy": dict(ood_policy or {}),
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
