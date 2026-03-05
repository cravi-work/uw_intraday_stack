from __future__ import annotations

import copy
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import yaml

from .adapt import get_adapt_support_status


_ENV_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)(:-([^}]*))?\}")


LEGACY_ALIAS_PAIRS = (
    ("system.timezone", "ingestion.timezone"),
    ("system.base_url", "network.base_url"),
)


UNSUPPORTED_CONFIG_KEYS = {
    "model.safe_mode_enabled": "Runtime does not honor model.safe_mode_enabled; remove it or wire the behavior before enabling it.",
}


VALID_ENDPOINT_PURPOSES = frozenset({
    "signal-critical",
    "context-only",
    "report-only",
    "disabled",
})

REQUIRED_ENDPOINT_POLICY_FIELDS = (
    "decision_path",
    "missing_affects_confidence",
    "stale_affects_confidence",
)

_ENDPOINT_PURPOSE_RUNTIME_CONTRACT = {
    "signal-critical": {
        "decision_path": True,
        "missing_affects_confidence": True,
        "stale_affects_confidence": True,
    },
    "context-only": {
        "decision_path": False,
        "missing_affects_confidence": False,
        "stale_affects_confidence": False,
    },
    "report-only": {
        "decision_path": False,
        "missing_affects_confidence": False,
        "stale_affects_confidence": False,
    },
    "disabled": {
        "decision_path": False,
        "missing_affects_confidence": False,
        "stale_affects_confidence": False,
    },
}

_PLAN_PURPOSES_BY_SECTION = {
    "default": frozenset({"signal-critical", "report-only", "disabled"}),
    "market_context": frozenset({"context-only", "disabled"}),
}

_DEFAULT_OOD_ASSESSMENT_POLICY = {
    "contract_version": "ood_assessment/v1",
    "degraded_coverage_threshold": 0.85,
    "out_coverage_threshold": 0.50,
    "boundary_slack": 1e-6,
    "require_assessment_before_emission": True,
}

_DEFAULT_OOD_PROBABILITY_POLICY = {
    "contract_version": "ood_probability/v1",
    "out_confidence_scale": 0.0,
    "out_emit_calibrated": False,
    "unknown_confidence_scale": 0.0,
    "unknown_emit_calibrated": False,
    "degraded_confidence_scale": 0.50,
    "degraded_emit_calibrated": True,
}

_DEFAULT_DECISION_PATH_POLICY = {
    "contract_version": "decision_path/v1",
    "zero_weight_is_non_decision": True,
    "require_feature_metadata": True,
    "allow_explicit_zero_weight_critical_override": True,
    "explicit_zero_weight_critical_features": {},
}

_DEFAULT_CALIBRATION_GOVERNANCE = {
    "contract_version": "calibration_registry/v1",
    "registry_version": None,
    "default_regime": "DEFAULT",
    "selection_policy": {
        "require_scope_match": True,
        "allow_legacy_fallback": False,
        "allow_generic_scope_fallback": True,
        "require_provenance": False,
        "required_provenance_fields": (
            "trained_from_utc",
            "trained_to_utc",
            "valid_from_utc",
            "valid_to_utc",
            "evidence_ref",
            "fit_sample_count",
        ),
        "institutional_grade": False,
    },
    "compatibility_rules": {
        "require_target_match": True,
        "require_horizon_match": True,
        "require_session_match": True,
        "require_regime_match": True,
        "require_replay_mode_match": True,
        "require_artifact_hash": True,
        "require_provenance_fields": False,
        "required_provenance_fields": (
            "trained_from_utc",
            "trained_to_utc",
            "valid_from_utc",
            "valid_to_utc",
            "evidence_ref",
            "fit_sample_count",
        ),
    },
    "artifacts": [],
}

VALID_GOVERNANCE_MODES = frozenset({"FORWARD_OBSERVATION", "INSTITUTIONAL_GRADE"})

_DEFAULT_RUNTIME_GOVERNANCE = {
    "governance_mode": "FORWARD_OBSERVATION",
}

_DEFAULT_OUTPUT_DOMAIN_POLICY = {
    "contract_version": "output_domain_policy/v1",
    "required_contract_version": "output_domain/v1",
    "require_bounded_output_contract": True,
    "require_expected_bounds": True,
    "require_emitted_units": True,
    "require_raw_input_units": True,
    "require_output_domain": True,
    "require_bounded_output_flag": True,
    "degrade_on_missing_contract": True,
    "enforce_on_decision_eligible_only": True,
}


@dataclass(frozen=True)
class AppConfig:
    raw: Dict[str, Any]


def _expand_env(value: str) -> str:
    def repl(m: re.Match) -> str:
        var = m.group(1)
        default = m.group(3) if m.group(2) else ""
        return os.getenv(var, default)
    return _ENV_PATTERN.sub(repl, value)



def _walk_expand(obj: Any) -> Any:
    if isinstance(obj, str):
        return _expand_env(obj)
    if isinstance(obj, list):
        return [_walk_expand(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _walk_expand(v) for k, v in obj.items()}
    return obj



def _ensure_mapping(cfg: Dict[str, Any], section: str) -> Dict[str, Any]:
    current = cfg.get(section)
    if current is None:
        cfg[section] = {}
        return cfg[section]
    if not isinstance(current, dict):
        raise ValueError(f"Config section '{section}' must be a mapping")
    return current



def _as_mapping(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}



def _normalize_str_sequence(value: Any) -> Tuple[str, ...]:
    if value in (None, ""):
        return tuple()
    if isinstance(value, str):
        raw = value.strip()
        return (raw,) if raw else tuple()
    if isinstance(value, Sequence):
        normalized: list[str] = []
        for item in value:
            raw = str(item or "").strip()
            if raw:
                normalized.append(raw)
        return tuple(normalized)
    raw = str(value).strip()
    return (raw,) if raw else tuple()



def _nested_get(mapping: Mapping[str, Any], path: str) -> Any:
    cur: Any = mapping
    for part in path.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            return None
        cur = cur[part]
    return cur



def _apply_legacy_aliases(cfg: Dict[str, Any]) -> Dict[str, Any]:
    system = _ensure_mapping(cfg, "system")
    ingestion = _ensure_mapping(cfg, "ingestion")
    network = _ensure_mapping(cfg, "network")

    if "timezone" in system:
        legacy = system["timezone"]
        modern = ingestion.get("timezone")
        if modern is not None and modern != legacy:
            raise ValueError("Conflicting config values for system.timezone and ingestion.timezone")
        ingestion.setdefault("timezone", legacy)

    if "base_url" in system:
        legacy = system["base_url"]
        modern = network.get("base_url")
        if modern is not None and modern != legacy:
            raise ValueError("Conflicting config values for system.base_url and network.base_url")
        network.setdefault("base_url", legacy)

    return cfg



def normalize_runtime_config(data: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(data, Mapping):
        raise ValueError("config root must be a mapping")
    cfg = copy.deepcopy(dict(data))
    for section in ("system", "ingestion", "network", "storage", "validation", "model"):
        if section in cfg and not isinstance(cfg[section], dict):
            raise ValueError(f"Config section '{section}' must be a mapping")
    return _apply_legacy_aliases(cfg)



def find_unsupported_config_keys(cfg: Mapping[str, Any]) -> Dict[str, str]:
    hits: Dict[str, str] = {}
    for path, message in UNSUPPORTED_CONFIG_KEYS.items():
        if _nested_get(cfg, path) is not None:
            hits[path] = message
    return hits



def resolve_decision_path_policy(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    normalized = normalize_runtime_config(cfg)
    validation = _as_mapping(normalized.get("validation"))
    policy = _as_mapping(validation.get("decision_path_policy"))
    explicit_overrides = _as_mapping(policy.get("explicit_zero_weight_critical_features"))
    return {
        "contract_version": str(policy.get("contract_version") or _DEFAULT_DECISION_PATH_POLICY["contract_version"]),
        "zero_weight_is_non_decision": bool(policy.get("zero_weight_is_non_decision", _DEFAULT_DECISION_PATH_POLICY["zero_weight_is_non_decision"])),
        "require_feature_metadata": bool(policy.get("require_feature_metadata", _DEFAULT_DECISION_PATH_POLICY["require_feature_metadata"])),
        "allow_explicit_zero_weight_critical_override": bool(
            policy.get(
                "allow_explicit_zero_weight_critical_override",
                _DEFAULT_DECISION_PATH_POLICY["allow_explicit_zero_weight_critical_override"],
            )
        ),
        "explicit_zero_weight_critical_features": {
            str(h): list(v)
            for h, v in explicit_overrides.items()
            if isinstance(v, list)
        },
    }



def resolve_ood_assessment_policy(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    normalized = normalize_runtime_config(cfg)
    model = _as_mapping(normalized.get("model"))
    raw_policy = _as_mapping(model.get("ood_assessment_policy"))
    if not raw_policy:
        raw_policy = _as_mapping(model.get("ood_policy")) or _as_mapping(model.get("ood")) or _as_mapping(normalized.get("ood_policy")) or _as_mapping(normalized.get("ood"))
    policy = dict(_DEFAULT_OOD_ASSESSMENT_POLICY)
    for key in ("contract_version", "require_assessment_before_emission"):
        if key in raw_policy:
            policy[key] = raw_policy[key]
    for key in ("degraded_coverage_threshold", "out_coverage_threshold", "boundary_slack"):
        if key in raw_policy:
            policy[key] = raw_policy[key]
    return policy



def resolve_ood_probability_policy(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    normalized = normalize_runtime_config(cfg)
    model = _as_mapping(normalized.get("model"))
    raw_policy = _as_mapping(model.get("ood_probability_policy")) or _as_mapping(model.get("ood_runtime_policy"))
    policy = dict(_DEFAULT_OOD_PROBABILITY_POLICY)
    for key, default in _DEFAULT_OOD_PROBABILITY_POLICY.items():
        if key in raw_policy:
            policy[key] = raw_policy[key]
    return policy



def resolve_calibration_governance(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    normalized = normalize_runtime_config(cfg)
    model = _as_mapping(normalized.get("model"))
    registry = _as_mapping(model.get("calibration_registry"))
    selection_policy = _as_mapping(registry.get("selection_policy"))
    compatibility_rules = _as_mapping(registry.get("compatibility_rules"))
    return {
        "contract_version": str(registry.get("contract_version") or _DEFAULT_CALIBRATION_GOVERNANCE["contract_version"]),
        "registry_version": registry.get("registry_version"),
        "default_regime": str(registry.get("default_regime") or _DEFAULT_CALIBRATION_GOVERNANCE["default_regime"]),
        "selection_policy": {
            "require_scope_match": bool(
                selection_policy.get(
                    "require_scope_match",
                    _DEFAULT_CALIBRATION_GOVERNANCE["selection_policy"]["require_scope_match"],
                )
            ),
            "allow_legacy_fallback": bool(
                selection_policy.get(
                    "allow_legacy_fallback",
                    _DEFAULT_CALIBRATION_GOVERNANCE["selection_policy"]["allow_legacy_fallback"],
                )
            ),
            "allow_generic_scope_fallback": bool(
                selection_policy.get(
                    "allow_generic_scope_fallback",
                    _DEFAULT_CALIBRATION_GOVERNANCE["selection_policy"]["allow_generic_scope_fallback"],
                )
            ),
            "require_provenance": bool(
                selection_policy.get(
                    "require_provenance",
                    _DEFAULT_CALIBRATION_GOVERNANCE["selection_policy"]["require_provenance"],
                )
            ),
            "required_provenance_fields": _normalize_str_sequence(
                selection_policy.get(
                    "required_provenance_fields",
                    _DEFAULT_CALIBRATION_GOVERNANCE["selection_policy"]["required_provenance_fields"],
                )
            ),
            "institutional_grade": bool(
                selection_policy.get(
                    "institutional_grade",
                    _DEFAULT_CALIBRATION_GOVERNANCE["selection_policy"]["institutional_grade"],
                )
            ),
        },
        "compatibility_rules": {
            **{
                key: bool(compatibility_rules.get(key, default))
                for key, default in _DEFAULT_CALIBRATION_GOVERNANCE["compatibility_rules"].items()
                if isinstance(default, bool)
            },
            "required_provenance_fields": _normalize_str_sequence(
                compatibility_rules.get(
                    "required_provenance_fields",
                    _DEFAULT_CALIBRATION_GOVERNANCE["compatibility_rules"]["required_provenance_fields"],
                )
            ),
        },
        "artifact_count": len(registry.get("artifacts") or []) if isinstance(registry.get("artifacts") or [], list) else 0,
        "legacy_calibration_declared": isinstance(model.get("calibration"), Mapping),
    }



def resolve_runtime_governance_mode(cfg: Mapping[str, Any]) -> str:
    normalized = normalize_runtime_config(cfg)
    validation = _as_mapping(normalized.get("validation"))
    raw = str(validation.get("governance_mode") or _DEFAULT_RUNTIME_GOVERNANCE["governance_mode"]).strip().upper()
    return raw or _DEFAULT_RUNTIME_GOVERNANCE["governance_mode"]



def resolve_output_domain_policy(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    normalized = normalize_runtime_config(cfg)
    validation = _as_mapping(normalized.get("validation"))
    raw_policy = _as_mapping(validation.get("output_domain_policy"))
    policy = dict(_DEFAULT_OUTPUT_DOMAIN_POLICY)
    for key in ("contract_version", "required_contract_version"):
        if key in raw_policy:
            policy[key] = str(raw_policy.get(key) or "").strip()
    for key in (
        "require_bounded_output_contract",
        "require_expected_bounds",
        "require_emitted_units",
        "require_raw_input_units",
        "require_output_domain",
        "require_bounded_output_flag",
        "degrade_on_missing_contract",
        "enforce_on_decision_eligible_only",
    ):
        if key in raw_policy:
            policy[key] = bool(raw_policy.get(key))
    return policy



def validate_governance_policy_config(cfg: Mapping[str, Any]) -> None:
    from .features import OUTPUT_DOMAIN_CONTRACT_VERSION

    normalized = normalize_runtime_config(cfg)
    validation = _as_mapping(normalized.get("validation"))
    model = _as_mapping(normalized.get("model"))

    governance_mode_explicit = "governance_mode" in validation
    governance_mode = resolve_runtime_governance_mode(normalized)
    if governance_mode_explicit and governance_mode not in VALID_GOVERNANCE_MODES:
        allowed = ", ".join(sorted(VALID_GOVERNANCE_MODES))
        raise ValueError(f"validation.governance_mode must be one of {allowed}")

    output_domain_section_explicit = "output_domain_policy" in validation
    output_domain_raw = validation.get("output_domain_policy")
    if output_domain_section_explicit and not isinstance(output_domain_raw, dict):
        raise ValueError("validation.output_domain_policy must be a mapping")
    output_domain_policy = resolve_output_domain_policy(normalized)
    if output_domain_section_explicit:
        if not str(output_domain_policy.get("contract_version") or "").strip():
            raise KeyError("Missing validation.output_domain_policy.contract_version")
        if not str(output_domain_policy.get("required_contract_version") or "").strip():
            raise KeyError("Missing validation.output_domain_policy.required_contract_version")
        required_output_domain_flags = (
            "require_bounded_output_contract",
            "require_expected_bounds",
            "require_emitted_units",
            "require_raw_input_units",
            "require_output_domain",
            "require_bounded_output_flag",
            "degrade_on_missing_contract",
            "enforce_on_decision_eligible_only",
        )
        for key in required_output_domain_flags:
            if key not in output_domain_raw:
                raise KeyError(f"Missing validation.output_domain_policy.{key}")
            if type(output_domain_raw[key]) is not bool:
                raise ValueError(f"validation.output_domain_policy.{key} must be a bool")
            if output_domain_policy[key] is not True:
                raise ValueError(f"validation.output_domain_policy.{key} must be true")
        if output_domain_policy["required_contract_version"] != OUTPUT_DOMAIN_CONTRACT_VERSION:
            raise ValueError(
                f"validation.output_domain_policy.required_contract_version must match runtime feature contract {OUTPUT_DOMAIN_CONTRACT_VERSION}"
            )

    calibration_registry = model.get("calibration_registry")
    if not isinstance(calibration_registry, dict):
        return
    selection_policy_raw = calibration_registry.get("selection_policy")
    compatibility_rules_raw = calibration_registry.get("compatibility_rules")
    if not isinstance(selection_policy_raw, dict):
        raise ValueError("model.calibration_registry.selection_policy must be a mapping")
    if not isinstance(compatibility_rules_raw, dict):
        raise ValueError("model.calibration_registry.compatibility_rules must be a mapping")

    calibration_governance = resolve_calibration_governance(normalized)
    selection_policy = calibration_governance["selection_policy"]
    compatibility_rules = calibration_governance["compatibility_rules"]

    new_selection_keys = {
        "allow_generic_scope_fallback",
        "require_provenance",
        "required_provenance_fields",
        "institutional_grade",
    }
    new_compatibility_keys = {
        "require_provenance_fields",
        "required_provenance_fields",
        "allow_generic_scope_fallback",
    }
    selection_has_new_fields = any(key in selection_policy_raw for key in new_selection_keys)
    compatibility_has_new_fields = any(key in compatibility_rules_raw for key in new_compatibility_keys)
    strict_calibration_governance = governance_mode == "INSTITUTIONAL_GRADE" or selection_has_new_fields or compatibility_has_new_fields

    if strict_calibration_governance:
        for key in ("allow_generic_scope_fallback", "require_provenance", "institutional_grade"):
            if key not in selection_policy_raw:
                raise KeyError(f"Missing model.calibration_registry.selection_policy.{key}")
            if type(selection_policy_raw[key]) is not bool:
                raise ValueError(f"model.calibration_registry.selection_policy.{key} must be a bool")
        if "required_provenance_fields" not in selection_policy_raw:
            raise KeyError("Missing model.calibration_registry.selection_policy.required_provenance_fields")
        if not isinstance(selection_policy_raw["required_provenance_fields"], list):
            raise ValueError("model.calibration_registry.selection_policy.required_provenance_fields must be a list")

        if "require_provenance_fields" not in compatibility_rules_raw:
            raise KeyError("Missing model.calibration_registry.compatibility_rules.require_provenance_fields")
        if type(compatibility_rules_raw["require_provenance_fields"]) is not bool:
            raise ValueError("model.calibration_registry.compatibility_rules.require_provenance_fields must be a bool")
        if "required_provenance_fields" not in compatibility_rules_raw:
            raise KeyError("Missing model.calibration_registry.compatibility_rules.required_provenance_fields")
        if not isinstance(compatibility_rules_raw["required_provenance_fields"], list):
            raise ValueError("model.calibration_registry.compatibility_rules.required_provenance_fields must be a list")
        if "allow_generic_scope_fallback" in compatibility_rules_raw and type(compatibility_rules_raw["allow_generic_scope_fallback"]) is not bool:
            raise ValueError("model.calibration_registry.compatibility_rules.allow_generic_scope_fallback must be a bool")

    required_provenance_fields = _normalize_str_sequence(selection_policy.get("required_provenance_fields"))
    compatibility_required_provenance_fields = _normalize_str_sequence(compatibility_rules.get("required_provenance_fields"))
    allowed_provenance_fields = set(_DEFAULT_CALIBRATION_GOVERNANCE["selection_policy"]["required_provenance_fields"])
    if strict_calibration_governance:
        if not required_provenance_fields:
            raise ValueError("model.calibration_registry.selection_policy.required_provenance_fields must not be empty")
        if not compatibility_required_provenance_fields:
            raise ValueError("model.calibration_registry.compatibility_rules.required_provenance_fields must not be empty")
        unknown_selection_fields = sorted(set(required_provenance_fields) - allowed_provenance_fields)
        if unknown_selection_fields:
            raise ValueError(
                "model.calibration_registry.selection_policy.required_provenance_fields contains unsupported fields: "
                f"{unknown_selection_fields}"
            )
        unknown_compat_fields = sorted(set(compatibility_required_provenance_fields) - allowed_provenance_fields)
        if unknown_compat_fields:
            raise ValueError(
                "model.calibration_registry.compatibility_rules.required_provenance_fields contains unsupported fields: "
                f"{unknown_compat_fields}"
            )
        if required_provenance_fields != compatibility_required_provenance_fields:
            raise ValueError(
                "model.calibration_registry.selection_policy.required_provenance_fields must match model.calibration_registry.compatibility_rules.required_provenance_fields"
            )
        if selection_policy["require_provenance"] and not compatibility_rules["require_provenance_fields"]:
            raise ValueError("model.calibration_registry.compatibility_rules.require_provenance_fields must be true when selection_policy.require_provenance is true")
        if "allow_generic_scope_fallback" in compatibility_rules_raw and bool(compatibility_rules_raw["allow_generic_scope_fallback"]) != bool(selection_policy["allow_generic_scope_fallback"]):
            raise ValueError(
                "model.calibration_registry.compatibility_rules.allow_generic_scope_fallback must match model.calibration_registry.selection_policy.allow_generic_scope_fallback"
            )

    if governance_mode == "INSTITUTIONAL_GRADE":
        if selection_policy["institutional_grade"] is not True:
            raise ValueError("model.calibration_registry.selection_policy.institutional_grade must be true when validation.governance_mode=INSTITUTIONAL_GRADE")
        if selection_policy["allow_generic_scope_fallback"] is not False:
            raise ValueError("model.calibration_registry.selection_policy.allow_generic_scope_fallback must be false in institutional-grade mode")
        if selection_policy["require_provenance"] is not True:
            raise ValueError("model.calibration_registry.selection_policy.require_provenance must be true in institutional-grade mode")
        if compatibility_rules["require_provenance_fields"] is not True:
            raise ValueError("model.calibration_registry.compatibility_rules.require_provenance_fields must be true in institutional-grade mode")
    elif strict_calibration_governance and selection_policy["institutional_grade"] is True:
        raise ValueError("model.calibration_registry.selection_policy.institutional_grade=true requires validation.governance_mode=INSTITUTIONAL_GRADE")

    artifacts = calibration_registry.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("model.calibration_registry.artifacts must be a non-empty list")
    if strict_calibration_governance and selection_policy["require_provenance"]:
        for idx, artifact in enumerate(artifacts):
            if not isinstance(artifact, Mapping):
                raise ValueError(f"model.calibration_registry.artifacts[{idx}] must be a mapping")
            provenance = _as_mapping(artifact.get("provenance"))
            for field_name in required_provenance_fields:
                value = provenance.get(field_name, artifact.get(field_name))
                if value in (None, ""):
                    raise KeyError(f"Missing model.calibration_registry.artifacts[{idx}].{field_name}")
            if (
                compatibility_rules.get("require_replay_mode_match")
                and not selection_policy.get("allow_generic_scope_fallback")
            ):
                scope = _as_mapping(artifact.get("scope"))
                replay_scope = str(scope.get("replay_mode") or "").strip().upper()
                if replay_scope in {"", "ANY"}:
                    raise ValueError(
                        f"model.calibration_registry.artifacts[{idx}].scope.replay_mode may not be ANY when replay-mode match is required and generic fallback is disabled"
                    )




def summarize_effective_runtime_config(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    normalized = normalize_runtime_config(cfg)
    ingestion = normalized.get("ingestion", {}) or {}
    network = normalized.get("network", {}) or {}
    system = normalized.get("system", {}) or {}
    validation = normalized.get("validation", {}) or {}
    model = normalized.get("model", {}) or {}
    target_spec = model.get("target_spec", {}) if isinstance(model.get("target_spec"), dict) else {}
    label_contract = validation.get("label_contract", {}) if isinstance(validation.get("label_contract"), dict) else {}
    criticals = validation.get("horizon_critical_features", {}) if isinstance(validation.get("horizon_critical_features"), dict) else {}
    adapt_status = get_adapt_support_status(normalized)
    decision_path_policy = resolve_decision_path_policy(normalized)
    ood_assessment_policy = resolve_ood_assessment_policy(normalized)
    ood_probability_policy = resolve_ood_probability_policy(normalized)
    calibration_governance = resolve_calibration_governance(normalized)
    governance_mode = resolve_runtime_governance_mode(normalized)
    output_domain_policy = resolve_output_domain_policy(normalized)

    return {
        "governance_mode": governance_mode,
        "timezone": ingestion.get("timezone") or ingestion.get("timezone_name") or ingestion.get("market_timezone"),
        "api_key_env": system.get("api_key_env"),
        "base_url": network.get("base_url"),
        "model_name": model.get("model_name") or "bounded_additive_score",
        "model_version": model.get("model_version") or "UNSPECIFIED",
        "target_name": target_spec.get("target_name") or target_spec.get("name"),
        "target_version": target_spec.get("target_version") or target_spec.get("version"),
        "threshold_policy_version": label_contract.get("threshold_policy_version") or label_contract.get("policy_version"),
        "label_version": label_contract.get("label_version"),
        "horizon_weights_source": validation.get("horizon_weights_source"),
        "critical_feature_policy": {str(k): list(v) for k, v in criticals.items() if isinstance(v, list)},
        "decision_path_policy_version": decision_path_policy["contract_version"],
        "zero_weight_is_non_decision": decision_path_policy["zero_weight_is_non_decision"],
        "require_feature_metadata": decision_path_policy["require_feature_metadata"],
        "zero_weight_critical_overrides": decision_path_policy["explicit_zero_weight_critical_features"],
        "ood_assessment_policy_version": str(ood_assessment_policy.get("contract_version") or ""),
        "ood_probability_policy_version": str(ood_probability_policy.get("contract_version") or ""),
        "ood_assessment_thresholds": {
            "degraded_coverage_threshold": float(ood_assessment_policy["degraded_coverage_threshold"]),
            "out_coverage_threshold": float(ood_assessment_policy["out_coverage_threshold"]),
            "boundary_slack": float(ood_assessment_policy["boundary_slack"]),
        },
        "calibration_registry_version": calibration_governance["registry_version"],
        "calibration_registry_contract_version": calibration_governance["contract_version"],
        "calibration_scope_required": calibration_governance["selection_policy"]["require_scope_match"],
        "calibration_allow_legacy_fallback": calibration_governance["selection_policy"]["allow_legacy_fallback"],
        "calibration_allow_generic_scope_fallback": calibration_governance["selection_policy"]["allow_generic_scope_fallback"],
        "calibration_require_provenance": calibration_governance["selection_policy"]["require_provenance"],
        "calibration_required_provenance_fields": list(calibration_governance["selection_policy"]["required_provenance_fields"]),
        "calibration_institutional_grade": calibration_governance["selection_policy"]["institutional_grade"],
        "calibration_compatibility_rules": calibration_governance["compatibility_rules"],
        "calibration_artifact_count": calibration_governance["artifact_count"],
        "legacy_calibration_declared": calibration_governance["legacy_calibration_declared"],
        "output_domain_policy_version": str(output_domain_policy.get("contract_version") or ""),
        "output_domain_required_contract_version": str(output_domain_policy.get("required_contract_version") or ""),
        "output_domain_requirements": {
            "require_bounded_output_contract": bool(output_domain_policy.get("require_bounded_output_contract")),
            "require_expected_bounds": bool(output_domain_policy.get("require_expected_bounds")),
            "require_emitted_units": bool(output_domain_policy.get("require_emitted_units")),
            "require_raw_input_units": bool(output_domain_policy.get("require_raw_input_units")),
            "require_output_domain": bool(output_domain_policy.get("require_output_domain")),
            "require_bounded_output_flag": bool(output_domain_policy.get("require_bounded_output_flag")),
            "degrade_on_missing_contract": bool(output_domain_policy.get("degrade_on_missing_contract")),
            "enforce_on_decision_eligible_only": bool(output_domain_policy.get("enforce_on_decision_eligible_only")),
        },
        "adapt_enabled": adapt_status.enabled_requested,
        "adapt_supported": adapt_status.supported,
        "adapt_rejection_reason": adapt_status.reason,
    }



def resolve_endpoint_purpose_contract(
    entry: Mapping[str, Any],
    *,
    section_name: str,
    entry_index: int,
    require_explicit: bool = True,
) -> Dict[str, Any]:
    if not isinstance(entry, Mapping):
        raise ValueError(f"endpoint_plan.yaml plans.{section_name}[{entry_index}] must be a mapping")

    name = str(entry.get("name", "")).strip()
    method = str(entry.get("method", "")).upper().strip()
    path = str(entry.get("path", "")).strip()
    purpose = str(entry.get("purpose", "")).strip()

    if not name:
        raise ValueError(f"endpoint_plan.yaml plans.{section_name}[{entry_index}] missing non-empty 'name'")
    if not method:
        raise ValueError(f"endpoint_plan.yaml plans.{section_name}[{entry_index}] missing non-empty 'method'")
    if not path:
        raise ValueError(f"endpoint_plan.yaml plans.{section_name}[{entry_index}] missing non-empty 'path'")
    if not purpose:
        if require_explicit:
            raise ValueError(f"endpoint_plan.yaml plans.{section_name}[{entry_index}] missing required 'purpose'")
        purpose = "context-only" if str(section_name) == "market_context" else "signal-critical"
    if purpose not in VALID_ENDPOINT_PURPOSES:
        allowed = ", ".join(sorted(VALID_ENDPOINT_PURPOSES))
        raise ValueError(
            f"endpoint_plan.yaml plans.{section_name}[{entry_index}] has invalid purpose '{purpose}'. Allowed: {allowed}"
        )

    allowed_for_section = _PLAN_PURPOSES_BY_SECTION.get(str(section_name), VALID_ENDPOINT_PURPOSES)
    if purpose not in allowed_for_section:
        allowed = ", ".join(sorted(allowed_for_section))
        raise ValueError(
            f"endpoint_plan.yaml plans.{section_name}[{entry_index}] purpose '{purpose}' is invalid for section '{section_name}'. "
            f"Allowed: {allowed}"
        )

    normalized = dict(entry)
    normalized["name"] = name
    normalized["method"] = method
    normalized["path"] = path
    normalized["purpose"] = purpose

    expected_contract = _ENDPOINT_PURPOSE_RUNTIME_CONTRACT[purpose]
    for field in REQUIRED_ENDPOINT_POLICY_FIELDS:
        if field not in normalized:
            if require_explicit:
                raise ValueError(
                    f"endpoint_plan.yaml plans.{section_name}[{entry_index}] missing required '{field}' for purpose '{purpose}'"
                )
            normalized[field] = expected_contract[field]
        value = normalized[field]
        if not isinstance(value, bool):
            raise ValueError(
                f"endpoint_plan.yaml plans.{section_name}[{entry_index}] field '{field}' must be a boolean"
            )
        expected_value = expected_contract[field]
        if value is not expected_value:
            raise ValueError(
                f"endpoint_plan.yaml plans.{section_name}[{entry_index}] field '{field}'={value!r} is incompatible with purpose '{purpose}'. "
                f"Expected {expected_value!r}."
            )

    normalized["path_params"] = normalized.get("path_params", {}) or {}
    normalized["query_params"] = normalized.get("query_params", {}) or {}
    normalized["purpose_contract_version"] = "v1"
    return normalized


def load_yaml(path: str | Path) -> AppConfig:
    p = Path(path)
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("config root must be a mapping")
    expanded = _walk_expand(data)
    normalized = normalize_runtime_config(expanded)
    return AppConfig(raw=normalized)



def _validate_endpoint_plan_contract(data: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(data, Mapping) or "plans" not in data:
        raise ValueError("endpoint_plan.yaml must contain a 'plans' mapping")

    plans = data.get("plans")
    if not isinstance(plans, Mapping):
        raise ValueError("endpoint_plan.yaml 'plans' must be a mapping")

    normalized: Dict[str, Any] = dict(data)
    normalized_plans: Dict[str, Any] = {}
    seen_names: set[tuple[str, str]] = set()
    seen_signatures: set[tuple[str, str, str]] = set()

    for section_name, entries in plans.items():
        if not isinstance(entries, list):
            raise ValueError(f"endpoint_plan.yaml plans.{section_name} must be a list")

        normalized_entries = []
        for idx, entry in enumerate(entries):
            normalized_entry = resolve_endpoint_purpose_contract(
                entry,
                section_name=str(section_name),
                entry_index=idx,
                require_explicit=True,
            )

            name_key = (str(section_name), normalized_entry["name"])
            if name_key in seen_names:
                raise ValueError(f"Duplicate endpoint plan name '{normalized_entry['name']}' in section '{section_name}'")
            seen_names.add(name_key)

            sig_key = (str(section_name), normalized_entry["method"], normalized_entry["path"])
            if sig_key in seen_signatures:
                raise ValueError(
                    f"Duplicate endpoint signature '{normalized_entry['method']} {normalized_entry['path']}' in section '{section_name}'"
                )
            seen_signatures.add(sig_key)
            normalized_entries.append(normalized_entry)

        normalized_plans[str(section_name)] = normalized_entries

    normalized["plans"] = normalized_plans
    return normalized



def load_endpoint_plan(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    return _validate_endpoint_plan_contract(data)
