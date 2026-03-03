from __future__ import annotations

import copy
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping

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

_PLAN_PURPOSES_BY_SECTION = {
    "default": frozenset({"signal-critical", "report-only", "disabled"}),
    "market_context": frozenset({"context-only", "disabled"}),
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



def summarize_effective_runtime_config(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    normalized = normalize_runtime_config(cfg)
    ingestion = normalized.get("ingestion", {}) or {}
    network = normalized.get("network", {}) or {}
    system = normalized.get("system", {}) or {}
    validation = normalized.get("validation", {}) or {}
    model = normalized.get("model", {}) or {}
    target_spec = model.get("target_spec", {}) if isinstance(model.get("target_spec"), dict) else {}
    calibration = model.get("calibration", {}) if isinstance(model.get("calibration"), dict) else {}
    label_contract = validation.get("label_contract", {}) if isinstance(validation.get("label_contract"), dict) else {}
    criticals = validation.get("horizon_critical_features", {}) if isinstance(validation.get("horizon_critical_features"), dict) else {}
    adapt_status = get_adapt_support_status(normalized)

    return {
        "timezone": ingestion.get("timezone") or ingestion.get("timezone_name") or ingestion.get("market_timezone"),
        "api_key_env": system.get("api_key_env"),
        "base_url": network.get("base_url"),
        "model_name": model.get("model_name") or "bounded_additive_score",
        "model_version": model.get("model_version") or "UNSPECIFIED",
        "target_name": target_spec.get("target_name") or target_spec.get("name"),
        "target_version": target_spec.get("target_version") or target_spec.get("version"),
        "calibration_artifact": calibration.get("artifact_name") or calibration.get("name"),
        "calibration_version": calibration.get("artifact_version") or calibration.get("version"),
        "threshold_policy_version": label_contract.get("threshold_policy_version") or label_contract.get("policy_version"),
        "label_version": label_contract.get("label_version"),
        "horizon_weights_source": validation.get("horizon_weights_source"),
        "critical_feature_policy": {str(k): list(v) for k, v in criticals.items() if isinstance(v, list)},
        "adapt_enabled": adapt_status.enabled_requested,
        "adapt_supported": adapt_status.supported,
        "adapt_rejection_reason": adapt_status.reason,
    }



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

    seen_names: set[tuple[str, str]] = set()
    seen_signatures: set[tuple[str, str, str]] = set()

    for section_name, entries in plans.items():
        if not isinstance(entries, list):
            raise ValueError(f"endpoint_plan.yaml plans.{section_name} must be a list")
        allowed_for_section = _PLAN_PURPOSES_BY_SECTION.get(str(section_name), VALID_ENDPOINT_PURPOSES)

        for idx, entry in enumerate(entries):
            if not isinstance(entry, Mapping):
                raise ValueError(f"endpoint_plan.yaml plans.{section_name}[{idx}] must be a mapping")

            name = str(entry.get("name", "")).strip()
            method = str(entry.get("method", "")).upper().strip()
            path = str(entry.get("path", "")).strip()
            purpose = str(entry.get("purpose", "")).strip()

            if not name:
                raise ValueError(f"endpoint_plan.yaml plans.{section_name}[{idx}] missing non-empty 'name'")
            if not method:
                raise ValueError(f"endpoint_plan.yaml plans.{section_name}[{idx}] missing non-empty 'method'")
            if not path:
                raise ValueError(f"endpoint_plan.yaml plans.{section_name}[{idx}] missing non-empty 'path'")
            if not purpose:
                raise ValueError(f"endpoint_plan.yaml plans.{section_name}[{idx}] missing required 'purpose'")
            if purpose not in VALID_ENDPOINT_PURPOSES:
                allowed = ", ".join(sorted(VALID_ENDPOINT_PURPOSES))
                raise ValueError(
                    f"endpoint_plan.yaml plans.{section_name}[{idx}] has invalid purpose '{purpose}'. Allowed: {allowed}"
                )
            if purpose not in allowed_for_section:
                allowed = ", ".join(sorted(allowed_for_section))
                raise ValueError(
                    f"endpoint_plan.yaml plans.{section_name}[{idx}] purpose '{purpose}' is invalid for section '{section_name}'. "
                    f"Allowed: {allowed}"
                )

            name_key = (str(section_name), name)
            if name_key in seen_names:
                raise ValueError(f"Duplicate endpoint plan name '{name}' in section '{section_name}'")
            seen_names.add(name_key)

            sig_key = (str(section_name), method, path)
            if sig_key in seen_signatures:
                raise ValueError(
                    f"Duplicate endpoint signature '{method} {path}' in section '{section_name}'"
                )
            seen_signatures.add(sig_key)

    return dict(data)



def load_endpoint_plan(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    return _validate_endpoint_plan_contract(data)
