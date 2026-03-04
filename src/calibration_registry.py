from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from .models import (
    CalibrationArtifactRef,
    HorizonKind,
    PredictionTargetSpec,
    ReplayMode,
    SessionState,
    build_calibration_artifact_ref,
)

ANY_SCOPE = "ANY"
DEFAULT_REGIME = "DEFAULT"


@dataclass(frozen=True)
class CalibrationSelectionRequest:
    target_name: str
    target_version: str
    horizon_kind: str
    horizon_minutes: Optional[int]
    session_state: str
    regime: str
    replay_mode: str
    contract_version: str = "calibration_request/v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_name": self.target_name,
            "target_version": self.target_version,
            "horizon_kind": self.horizon_kind,
            "horizon_minutes": self.horizon_minutes,
            "session_state": self.session_state,
            "regime": self.regime,
            "replay_mode": self.replay_mode,
            "contract_version": self.contract_version,
        }


@dataclass(frozen=True)
class CalibrationRegistry:
    artifacts: Tuple[CalibrationArtifactRef, ...]
    registry_version: str
    registry_source: str
    invalid_entries: Tuple[str, ...] = field(default_factory=tuple)
    default_regime: str = DEFAULT_REGIME
    selection_policy: Dict[str, Any] = field(default_factory=dict)
    compatibility_rules: Dict[str, Any] = field(default_factory=dict)
    contract_version: str = "calibration_registry/v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "registry_version": self.registry_version,
            "registry_source": self.registry_source,
            "artifact_count": len(self.artifacts),
            "invalid_entries": list(self.invalid_entries),
            "default_regime": self.default_regime,
            "selection_policy": dict(self.selection_policy),
            "compatibility_rules": dict(self.compatibility_rules),
            "contract_version": self.contract_version,
        }


@dataclass(frozen=True)
class CalibrationSelectionResult:
    artifact: Optional[CalibrationArtifactRef]
    request: CalibrationSelectionRequest
    reason_code: str
    reasons: Tuple[str, ...]
    registry_version: str
    registry_source: str
    candidate_count: int
    compatible_candidate_count: int
    invalid_entries: Tuple[str, ...] = field(default_factory=tuple)
    selection_policy: Dict[str, Any] = field(default_factory=dict)
    compatibility_rules: Dict[str, Any] = field(default_factory=dict)
    contract_version: str = "calibration_selection/v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reason_code": self.reason_code,
            "reasons": list(self.reasons),
            "request": self.request.to_dict(),
            "registry_version": self.registry_version,
            "registry_source": self.registry_source,
            "candidate_count": self.candidate_count,
            "compatible_candidate_count": self.compatible_candidate_count,
            "invalid_entries": list(self.invalid_entries),
            "selection_policy": dict(self.selection_policy),
            "compatibility_rules": dict(self.compatibility_rules),
            "artifact": self.artifact.to_dict() if self.artifact is not None else None,
            "artifact_hash": self.artifact.artifact_hash if self.artifact is not None else None,
            "calibration_scope": self.artifact.calibration_scope if self.artifact is not None else None,
            "contract_version": self.contract_version,
        }


def _as_mapping(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _normalize_text(value: Any, *, default: str = "") -> str:
    raw = str(value or "").strip()
    return raw if raw else default


def _normalize_scope_token(value: Any, *, default: str = ANY_SCOPE) -> str:
    raw = _normalize_text(value, default=default)
    return raw.upper() if raw else default


def _normalize_session_state(value: Any) -> str:
    if isinstance(value, SessionState):
        return value.value
    raw = _normalize_scope_token(value, default=SessionState.CLOSED.value)
    if raw in {SessionState.PREMARKET.value, SessionState.RTH.value, SessionState.AFTERHOURS.value, SessionState.CLOSED.value}:
        return raw
    return SessionState.CLOSED.value


def _normalize_horizon_kind(value: Any) -> str:
    if isinstance(value, HorizonKind):
        return value.value
    raw = _normalize_scope_token(value, default=ANY_SCOPE)
    if raw in {ANY_SCOPE, HorizonKind.FIXED.value, HorizonKind.TO_CLOSE.value}:
        return raw
    return ANY_SCOPE


def _normalize_replay_mode(value: Any) -> str:
    if isinstance(value, ReplayMode):
        return value.value
    raw = _normalize_scope_token(value, default=ReplayMode.UNKNOWN.value)
    if raw in {ANY_SCOPE, ReplayMode.UNKNOWN.value, ReplayMode.LIVE_LIKE_OBSERVED.value, ReplayMode.RESEARCH_RESTATED.value}:
        return raw
    return ReplayMode.UNKNOWN.value


def _coerce_optional_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    return int(value)


def _scope_specificity(artifact: CalibrationArtifactRef) -> tuple[int, str, str]:
    score = 0
    if str(artifact.scope_horizon_kind).upper() != ANY_SCOPE:
        score += 10
    if artifact.scope_horizon_minutes is not None:
        score += 20
    if str(artifact.scope_session).upper() != ANY_SCOPE:
        score += 5
    if str(artifact.scope_regime).upper() != ANY_SCOPE:
        score += 3
    if str(artifact.scope_replay_mode).upper() != ANY_SCOPE:
        score += 1
    return (-score, str(artifact.artifact_name), str(artifact.artifact_version))


def _artifact_matches_target(artifact: CalibrationArtifactRef, request: CalibrationSelectionRequest) -> bool:
    return artifact.target_name == request.target_name and artifact.target_version == request.target_version


def _artifact_matches_horizon(artifact: CalibrationArtifactRef, request: CalibrationSelectionRequest) -> bool:
    scope_kind = _normalize_horizon_kind(artifact.scope_horizon_kind)
    if scope_kind == ANY_SCOPE:
        return True
    if scope_kind != request.horizon_kind:
        return False
    if scope_kind == HorizonKind.FIXED.value and artifact.scope_horizon_minutes is not None:
        if request.horizon_minutes is None:
            return False
        return int(artifact.scope_horizon_minutes) == int(request.horizon_minutes)
    if scope_kind == HorizonKind.TO_CLOSE.value and artifact.scope_horizon_minutes not in (None, 0):
        return False
    return True


def _artifact_matches_session(artifact: CalibrationArtifactRef, request: CalibrationSelectionRequest) -> bool:
    scope_session = _normalize_scope_token(artifact.scope_session)
    return scope_session == ANY_SCOPE or scope_session == request.session_state


def _artifact_matches_regime(artifact: CalibrationArtifactRef, request: CalibrationSelectionRequest) -> bool:
    scope_regime = _normalize_scope_token(artifact.scope_regime)
    request_regime = _normalize_scope_token(request.regime, default=DEFAULT_REGIME)
    return scope_regime == ANY_SCOPE or scope_regime == request_regime


def _artifact_matches_replay_mode(artifact: CalibrationArtifactRef, request: CalibrationSelectionRequest) -> bool:
    scope_replay = _normalize_replay_mode(artifact.scope_replay_mode)
    return scope_replay == ANY_SCOPE or scope_replay == request.replay_mode


def _build_registry_artifact(
    entry: Mapping[str, Any],
    *,
    model_name: str,
    default_target_spec: Optional[PredictionTargetSpec],
    source_name: str,
) -> CalibrationArtifactRef:
    cfg = dict(entry)
    scope_cfg = _as_mapping(cfg.get("scope"))

    bins_raw = cfg.get("bins")
    mapped_raw = cfg.get("mapped")
    if not isinstance(bins_raw, Sequence) or isinstance(bins_raw, str):
        raise ValueError("missing bins")
    if not isinstance(mapped_raw, Sequence) or isinstance(mapped_raw, str):
        raise ValueError("missing mapped")

    try:
        bins = tuple(float(x) for x in bins_raw)
        mapped = tuple(float(x) for x in mapped_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("non-numeric bins/mapped") from exc

    target_name = _normalize_text(cfg.get("target_name"), default=(default_target_spec.target_name if default_target_spec else ""))
    target_version = _normalize_text(cfg.get("target_version"), default=(default_target_spec.target_version if default_target_spec else ""))

    artifact = CalibrationArtifactRef(
        artifact_name=_normalize_text(cfg.get("artifact_name"), default=f"{model_name}_calibration"),
        artifact_version=_normalize_text(cfg.get("artifact_version") or cfg.get("version")),
        target_name=target_name,
        target_version=target_version,
        bins=bins,
        mapped=mapped,
        artifact_source=_normalize_text(cfg.get("artifact_source"), default=source_name),
        scope_horizon_kind=_normalize_horizon_kind(scope_cfg.get("horizon_kind")),
        scope_horizon_minutes=_coerce_optional_int(scope_cfg.get("horizon_minutes")),
        scope_session=_normalize_scope_token(scope_cfg.get("session"), default=ANY_SCOPE),
        scope_regime=_normalize_scope_token(scope_cfg.get("regime"), default=ANY_SCOPE),
        scope_replay_mode=_normalize_replay_mode(scope_cfg.get("replay_mode")),
        scope_contract_version=_normalize_text(scope_cfg.get("scope_contract_version"), default="calibration_scope/v1"),
    )
    if not artifact.is_valid():
        raise ValueError("invalid calibration artifact")
    return artifact


def build_calibration_registry(
    model_cfg: Optional[Mapping[str, Any]],
    *,
    target_spec: Optional[PredictionTargetSpec],
) -> CalibrationRegistry:
    cfg = dict(model_cfg or {})
    model_name = _normalize_text(cfg.get("model_name"), default="model")
    registry_cfg = _as_mapping(cfg.get("calibration_registry"))

    if registry_cfg:
        artifacts_raw = registry_cfg.get("artifacts") or []
        artifacts: list[CalibrationArtifactRef] = []
        invalid_entries: list[str] = []
        if not isinstance(artifacts_raw, Sequence) or isinstance(artifacts_raw, str):
            invalid_entries.append("artifacts: not-a-list")
            artifacts_raw = []
        for idx, entry in enumerate(artifacts_raw):
            if not isinstance(entry, Mapping):
                invalid_entries.append(f"artifacts[{idx}]: not-a-mapping")
                continue
            try:
                artifacts.append(
                    _build_registry_artifact(
                        entry,
                        model_name=model_name,
                        default_target_spec=target_spec,
                        source_name="calibration_registry",
                    )
                )
            except Exception as exc:  # explicit parse rejection surfaced in selection diagnostics
                invalid_entries.append(f"artifacts[{idx}]: {exc}")
        return CalibrationRegistry(
            artifacts=tuple(artifacts),
            registry_version=_normalize_text(registry_cfg.get("registry_version"), default="runtime_v1"),
            registry_source="calibration_registry",
            invalid_entries=tuple(invalid_entries),
            default_regime=_normalize_scope_token(registry_cfg.get("default_regime"), default=DEFAULT_REGIME),
            selection_policy=_as_mapping(registry_cfg.get("selection_policy")),
            compatibility_rules=_as_mapping(registry_cfg.get("compatibility_rules")),
            contract_version=_normalize_text(registry_cfg.get("contract_version"), default="calibration_registry/v1"),
        )

    legacy_ref = build_calibration_artifact_ref(cfg, target_spec=target_spec)
    if legacy_ref is not None:
        return CalibrationRegistry(
            artifacts=(legacy_ref,),
            registry_version=str(legacy_ref.artifact_version),
            registry_source="legacy_model_calibration",
            invalid_entries=tuple(),
            default_regime=DEFAULT_REGIME,
            selection_policy={"legacy_unscoped": True},
            compatibility_rules={},
            contract_version="calibration_registry/v1",
        )

    return CalibrationRegistry(
        artifacts=tuple(),
        registry_version="UNCONFIGURED",
        registry_source="none",
        invalid_entries=tuple(),
        default_regime=DEFAULT_REGIME,
        selection_policy={},
        compatibility_rules={},
        contract_version="calibration_registry/v1",
    )


def resolve_calibration_regime(model_cfg: Optional[Mapping[str, Any]]) -> str:
    cfg = dict(model_cfg or {})
    registry_cfg = _as_mapping(cfg.get("calibration_registry"))
    explicit = registry_cfg.get("runtime_regime") or registry_cfg.get("default_regime")
    return _normalize_scope_token(explicit, default=DEFAULT_REGIME)


def select_calibration_artifact(
    model_cfg: Optional[Mapping[str, Any]],
    *,
    target_spec: Optional[PredictionTargetSpec],
    horizon_kind: Any,
    horizon_minutes: Optional[int],
    session_state: Any,
    regime: Optional[str] = None,
    replay_mode: Any = ReplayMode.UNKNOWN,
) -> CalibrationSelectionResult:
    request = CalibrationSelectionRequest(
        target_name=str(target_spec.target_name) if target_spec is not None else "",
        target_version=str(target_spec.target_version) if target_spec is not None else "",
        horizon_kind=_normalize_horizon_kind(horizon_kind),
        horizon_minutes=_coerce_optional_int(horizon_minutes),
        session_state=_normalize_session_state(session_state),
        regime=_normalize_scope_token(regime, default=DEFAULT_REGIME),
        replay_mode=_normalize_replay_mode(replay_mode),
    )

    registry = build_calibration_registry(model_cfg, target_spec=target_spec)
    all_artifacts = tuple(artifact for artifact in registry.artifacts if artifact.is_valid())

    if target_spec is None:
        return CalibrationSelectionResult(
            artifact=None,
            request=request,
            reason_code="MISSING_TARGET_SPEC",
            reasons=("missing_target_spec",),
            registry_version=registry.registry_version,
            registry_source=registry.registry_source,
            candidate_count=len(all_artifacts),
            compatible_candidate_count=0,
            invalid_entries=registry.invalid_entries,
            selection_policy=registry.selection_policy,
            compatibility_rules=registry.compatibility_rules,
        )

    if not all_artifacts:
        return CalibrationSelectionResult(
            artifact=None,
            request=request,
            reason_code="INVALID_REGISTRY_CONFIGURATION" if registry.invalid_entries else "NO_ARTIFACTS_CONFIGURED",
            reasons=tuple(registry.invalid_entries) if registry.invalid_entries else ("no_artifacts_configured",),
            registry_version=registry.registry_version,
            registry_source=registry.registry_source,
            candidate_count=0,
            compatible_candidate_count=0,
            invalid_entries=registry.invalid_entries,
            selection_policy=registry.selection_policy,
            compatibility_rules=registry.compatibility_rules,
        )

    target_candidates = tuple(a for a in all_artifacts if _artifact_matches_target(a, request))
    if not target_candidates:
        return CalibrationSelectionResult(
            artifact=None,
            request=request,
            reason_code="TARGET_MISMATCH",
            reasons=("target_mismatch",),
            registry_version=registry.registry_version,
            registry_source=registry.registry_source,
            candidate_count=len(all_artifacts),
            compatible_candidate_count=0,
            invalid_entries=registry.invalid_entries,
            selection_policy=registry.selection_policy,
            compatibility_rules=registry.compatibility_rules,
        )

    horizon_candidates = tuple(a for a in target_candidates if _artifact_matches_horizon(a, request))
    if not horizon_candidates:
        return CalibrationSelectionResult(
            artifact=None,
            request=request,
            reason_code="HORIZON_MISMATCH",
            reasons=("horizon_mismatch",),
            registry_version=registry.registry_version,
            registry_source=registry.registry_source,
            candidate_count=len(target_candidates),
            compatible_candidate_count=0,
            invalid_entries=registry.invalid_entries,
            selection_policy=registry.selection_policy,
            compatibility_rules=registry.compatibility_rules,
        )

    session_candidates = tuple(a for a in horizon_candidates if _artifact_matches_session(a, request))
    if not session_candidates:
        return CalibrationSelectionResult(
            artifact=None,
            request=request,
            reason_code="SESSION_MISMATCH",
            reasons=("session_mismatch",),
            registry_version=registry.registry_version,
            registry_source=registry.registry_source,
            candidate_count=len(horizon_candidates),
            compatible_candidate_count=0,
            invalid_entries=registry.invalid_entries,
            selection_policy=registry.selection_policy,
            compatibility_rules=registry.compatibility_rules,
        )

    regime_candidates = tuple(a for a in session_candidates if _artifact_matches_regime(a, request))
    if not regime_candidates:
        return CalibrationSelectionResult(
            artifact=None,
            request=request,
            reason_code="REGIME_MISMATCH",
            reasons=("regime_mismatch",),
            registry_version=registry.registry_version,
            registry_source=registry.registry_source,
            candidate_count=len(session_candidates),
            compatible_candidate_count=0,
            invalid_entries=registry.invalid_entries,
            selection_policy=registry.selection_policy,
            compatibility_rules=registry.compatibility_rules,
        )

    replay_candidates = tuple(a for a in regime_candidates if _artifact_matches_replay_mode(a, request))
    if not replay_candidates:
        return CalibrationSelectionResult(
            artifact=None,
            request=request,
            reason_code="REPLAY_MODE_MISMATCH",
            reasons=("replay_mode_mismatch",),
            registry_version=registry.registry_version,
            registry_source=registry.registry_source,
            candidate_count=len(regime_candidates),
            compatible_candidate_count=0,
            invalid_entries=registry.invalid_entries,
            selection_policy=registry.selection_policy,
            compatibility_rules=registry.compatibility_rules,
        )

    ordered = sorted(replay_candidates, key=_scope_specificity)
    selected = ordered[0]
    reasons = ["selected_scope_compatible_artifact"]
    if len(ordered) > 1:
        reasons.append(f"multiple_compatible_candidates:{len(ordered)}")

    return CalibrationSelectionResult(
        artifact=selected,
        request=request,
        reason_code="SELECTED",
        reasons=tuple(reasons),
        registry_version=registry.registry_version,
        registry_source=registry.registry_source,
        candidate_count=len(all_artifacts),
        compatible_candidate_count=len(replay_candidates),
        invalid_entries=registry.invalid_entries,
        selection_policy=registry.selection_policy,
        compatibility_rules=registry.compatibility_rules,
    )
