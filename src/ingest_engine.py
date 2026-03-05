# src/ingest_engine.py
from __future__ import annotations

import asyncio
import datetime as dt
import logging
import math
import uuid
import hashlib # CL-06: Deterministic generation
import json    # FIX: Added missing json import for diagnostics
from dataclasses import dataclass, asdict, replace
from typing import Any, Dict, List, Optional, Tuple

from .api_catalog_loader import load_api_catalog
from .adapt import validate_adapt_config
from .config_loader import (
    find_unsupported_config_keys,
    load_endpoint_plan,
    normalize_runtime_config,
    resolve_calibration_governance,
    resolve_decision_path_policy,
    resolve_endpoint_purpose_contract,
    resolve_ood_assessment_policy,
    resolve_ood_probability_policy,
    summarize_effective_runtime_config,
    validate_governance_policy_config,
)
from .file_lock import FileLock, FileLockError
from .logging_config import log_prediction_decision, structured_log
from .scheduler import ET, UTC, coerce_session_state, floor_to_interval, get_market_hours
from .storage import DbWriter
from .uw_client import UwClient
from .endpoint_rules import EmptyPayloadPolicy, validate_plan_coverage
from .features import extract_all
from .models import (
    bounded_additive_score,
    Prediction,
    DecisionGate,
    DataQualityState,
    RiskGateStatus,
    SignalState,
    SessionState,
    ConfidenceState,
    KNOWN_FEATURE_KEYS,
    build_prediction_target_spec,
    build_label_contract_spec,
    OODState,
    ReplayMode,
)
from .endpoint_truth import (
    EndpointContext,
    EndpointPayloadClass, 
    FreshnessState, 
    MetaContract,
    NaReasonCode,
    PayloadAssessment,
    classify_payload, 
    infer_source_time_hints,
    resolve_effective_payload, 
    to_utc_dt
)
from .freshness_policy import (
    FeatureDecisionReason,
    FeaturePolicyAssessment,
    assess_feature_freshness,
)
from .ood import OODAssessment, assess_operational_ood, resolve_ood_policy
from .calibration_registry import CalibrationSelectionResult, resolve_calibration_regime, select_calibration_artifact

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class PlannedCall:
    name: str
    method: str
    path: str
    path_params: Dict[str, Any]
    query_params: Dict[str, Any]
    is_market: bool
    purpose: str = "signal-critical"
    decision_path: bool = True
    missing_affects_confidence: bool = True
    stale_affects_confidence: bool = True
    purpose_contract_version: str = "v1"


def _require_mapping(mapping: Dict[str, Any], key: str, path: str) -> Dict[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, dict):
        raise KeyError(f"Missing {path}")
    return value


def _require_nonempty_str(mapping: Dict[str, Any], key: str, path: str) -> str:
    value = mapping.get(key)
    if value is None or not str(value).strip():
        raise KeyError(f"Missing {path}")
    return str(value).strip()


def _require_bool(mapping: Dict[str, Any], key: str, path: str) -> bool:
    if key not in mapping or type(mapping[key]) is not bool:
        raise KeyError(f"Missing {path}")
    return bool(mapping[key])


def _require_float(mapping: Dict[str, Any], key: str, path: str, *, minimum: Optional[float] = None, maximum: Optional[float] = None) -> float:
    if key not in mapping:
        raise KeyError(f"Missing {path}")
    value = mapping[key]
    if not isinstance(value, (float, int)) or not math.isfinite(float(value)):
        raise ValueError(f"{path} must be a finite float")
    value_f = float(value)
    if minimum is not None and value_f < minimum:
        raise ValueError(f"{path} must be >= {minimum}")
    if maximum is not None and value_f > maximum:
        raise ValueError(f"{path} must be <= {maximum}")
    return value_f


def _normalize_prediction_replay_mode(value: Any) -> ReplayMode:
    """
    Resolve the runtime replay/data-observation mode that should govern calibration
    selection and probability emission.

    Forward/non-replay prediction paths default to LIVE_LIKE_OBSERVED semantics so
    artifact selection does not silently collapse into UNKNOWN.
    """
    if value in (None, "", ReplayMode.UNKNOWN, ReplayMode.UNKNOWN.value):
        return ReplayMode.LIVE_LIKE_OBSERVED
    if isinstance(value, ReplayMode):
        return value
    try:
        mode = ReplayMode(str(value).upper())
    except Exception as exc:
        raise ValueError(f"Unsupported replay_mode for prediction generation: {value}") from exc
    return ReplayMode.LIVE_LIKE_OBSERVED if mode == ReplayMode.UNKNOWN else mode


def _effective_weights_for_horizon(cfg: Dict[str, Any], horizon: str) -> Dict[str, float]:
    val_cfg = cfg.get("validation", {}) or {}
    src = val_cfg.get("horizon_weights_source")
    if src == "model":
        base = dict((cfg.get("model", {}) or {}).get("weights", {}) or {})
        overrides = dict((val_cfg.get("horizon_weights_overrides", {}) or {}).get(horizon, {}) or {})
        base.update(overrides)
        return {str(k): float(v) for k, v in base.items()}
    return {str(k): float(v) for k, v in dict((val_cfg.get("horizon_weights", {}) or {}).get(horizon, {}) or {}).items()}


def _validate_governance_contract(cfg: Dict[str, Any], horizons: List[str]) -> None:
    val_cfg = cfg.get("validation", {}) or {}
    model_cfg = cfg.get("model", {}) or {}

    decision_path_policy = resolve_decision_path_policy(cfg)
    dp_section = val_cfg.get("decision_path_policy")
    if dp_section is not None and not isinstance(dp_section, dict):
        raise ValueError("validation.decision_path_policy must be a mapping")

    if isinstance(dp_section, dict):
        _require_nonempty_str(dp_section, "contract_version", "validation.decision_path_policy.contract_version")
        _require_bool(dp_section, "zero_weight_is_non_decision", "validation.decision_path_policy.zero_weight_is_non_decision")
        _require_bool(dp_section, "require_feature_metadata", "validation.decision_path_policy.require_feature_metadata")
        allow_zero_weight_override = _require_bool(
            dp_section,
            "allow_explicit_zero_weight_critical_override",
            "validation.decision_path_policy.allow_explicit_zero_weight_critical_override",
        )
        overrides_raw = _require_mapping(
            dp_section,
            "explicit_zero_weight_critical_features",
            "validation.decision_path_policy.explicit_zero_weight_critical_features",
        )
        for horizon in horizons:
            if horizon not in overrides_raw:
                raise KeyError(
                    f"Missing validation.decision_path_policy.explicit_zero_weight_critical_features for horizon '{horizon}'"
                )
            if not isinstance(overrides_raw[horizon], list):
                raise ValueError(
                    f"validation.decision_path_policy.explicit_zero_weight_critical_features['{horizon}'] must be a list"
                )
    else:
        allow_zero_weight_override = bool(decision_path_policy["allow_explicit_zero_weight_critical_override"])
        overrides_raw = {horizon: [] for horizon in horizons}

    if decision_path_policy["zero_weight_is_non_decision"] is not True:
        raise ValueError("validation.decision_path_policy.zero_weight_is_non_decision must be true")
    if decision_path_policy["require_feature_metadata"] is not True:
        raise ValueError("validation.decision_path_policy.require_feature_metadata must be true")

    for horizon in horizons:
        effective_weights = _effective_weights_for_horizon(cfg, horizon)
        zero_weight_keys = {k for k, v in effective_weights.items() if float(v) == 0.0}
        criticals = set(val_cfg.get("horizon_critical_features", {}).get(horizon, []) or [])
        overrides = set(overrides_raw.get(horizon, []) or [])
        invalid_override_targets = overrides - set(effective_weights.keys())
        if invalid_override_targets:
            raise ValueError(
                f"validation.decision_path_policy explicit override references unknown weighted features for horizon '{horizon}': {sorted(invalid_override_targets)}"
            )
        nonzero_override_targets = overrides - zero_weight_keys
        if nonzero_override_targets:
            raise ValueError(
                f"validation.decision_path_policy explicit override references non-zero-weight features for horizon '{horizon}': {sorted(nonzero_override_targets)}"
            )
        leaking_zero_weight_criticals = zero_weight_keys & criticals
        if leaking_zero_weight_criticals and not allow_zero_weight_override:
            raise ValueError(
                f"Zero-weight critical features are not allowed by validation.decision_path_policy for horizon '{horizon}': {sorted(leaking_zero_weight_criticals)}"
            )
        missing_explicit_override = leaking_zero_weight_criticals - overrides
        if missing_explicit_override:
            raise ValueError(
                f"Zero-weight critical features require explicit override in validation.decision_path_policy for horizon '{horizon}': {sorted(missing_explicit_override)}"
            )

    ood_assessment_raw = model_cfg.get("ood_assessment_policy")
    ood_assessment_policy = resolve_ood_assessment_policy(cfg)
    if ood_assessment_raw is not None:
        if not isinstance(ood_assessment_raw, dict):
            raise ValueError("model.ood_assessment_policy must be a mapping")
        _require_nonempty_str(ood_assessment_raw, "contract_version", "model.ood_assessment_policy.contract_version")
        _require_bool(ood_assessment_raw, "require_assessment_before_emission", "model.ood_assessment_policy.require_assessment_before_emission")
        degraded_threshold = _require_float(
            ood_assessment_raw, "degraded_coverage_threshold", "model.ood_assessment_policy.degraded_coverage_threshold", minimum=0.0, maximum=1.0
        )
        out_threshold = _require_float(
            ood_assessment_raw, "out_coverage_threshold", "model.ood_assessment_policy.out_coverage_threshold", minimum=0.0, maximum=1.0
        )
        _require_float(ood_assessment_raw, "boundary_slack", "model.ood_assessment_policy.boundary_slack", minimum=0.0)
        if out_threshold > degraded_threshold:
            raise ValueError("model.ood_assessment_policy.out_coverage_threshold must be <= degraded_coverage_threshold")
    if bool(ood_assessment_policy.get("require_assessment_before_emission", False)) is not True:
        raise ValueError("model.ood_assessment_policy.require_assessment_before_emission must be true")

    ood_probability_raw = model_cfg.get("ood_probability_policy")
    ood_probability_policy = resolve_ood_probability_policy(cfg)
    if ood_probability_raw is not None:
        if not isinstance(ood_probability_raw, dict):
            raise ValueError("model.ood_probability_policy must be a mapping")
        _require_nonempty_str(ood_probability_raw, "contract_version", "model.ood_probability_policy.contract_version")
        for field in ("out_confidence_scale", "unknown_confidence_scale", "degraded_confidence_scale"):
            _require_float(ood_probability_raw, field, f"model.ood_probability_policy.{field}", minimum=0.0, maximum=1.0)
        for field in ("out_emit_calibrated", "unknown_emit_calibrated", "degraded_emit_calibrated"):
            _require_bool(ood_probability_raw, field, f"model.ood_probability_policy.{field}")
    if bool(ood_probability_policy.get("out_emit_calibrated")):
        raise ValueError("model.ood_probability_policy.out_emit_calibrated must be false")
    if bool(ood_probability_policy.get("unknown_emit_calibrated")):
        raise ValueError("model.ood_probability_policy.unknown_emit_calibrated must be false")

    calibration_registry_raw = model_cfg.get("calibration_registry")
    calibration_governance = resolve_calibration_governance(cfg)
    if calibration_registry_raw is not None:
        if not isinstance(calibration_registry_raw, dict):
            raise ValueError("model.calibration_registry must be a mapping")
        _require_nonempty_str(calibration_registry_raw, "contract_version", "model.calibration_registry.contract_version")
        _require_nonempty_str(calibration_registry_raw, "registry_version", "model.calibration_registry.registry_version")
        _require_nonempty_str(calibration_registry_raw, "default_regime", "model.calibration_registry.default_regime")
        selection_policy_raw = _require_mapping(calibration_registry_raw, "selection_policy", "model.calibration_registry.selection_policy")
        compatibility_rules_raw = _require_mapping(calibration_registry_raw, "compatibility_rules", "model.calibration_registry.compatibility_rules")
        _require_bool(selection_policy_raw, "require_scope_match", "model.calibration_registry.selection_policy.require_scope_match")
        _require_bool(selection_policy_raw, "allow_legacy_fallback", "model.calibration_registry.selection_policy.allow_legacy_fallback")
        if calibration_governance["selection_policy"]["require_scope_match"] is not True:
            raise ValueError("model.calibration_registry.selection_policy.require_scope_match must be true")
        if calibration_governance["selection_policy"]["allow_legacy_fallback"] is not False:
            raise ValueError("model.calibration_registry.selection_policy.allow_legacy_fallback must be false")
        required_compatibility_rules = (
            "require_target_match",
            "require_horizon_match",
            "require_session_match",
            "require_regime_match",
            "require_replay_mode_match",
            "require_artifact_hash",
        )
        for key in required_compatibility_rules:
            _require_bool(compatibility_rules_raw, key, f"model.calibration_registry.compatibility_rules.{key}")
            if calibration_governance["compatibility_rules"][key] is not True:
                raise ValueError(f"model.calibration_registry.compatibility_rules.{key} must be true")
        artifacts = calibration_registry_raw.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            raise ValueError("model.calibration_registry.artifacts must be a non-empty list")
        if model_cfg.get("calibration") is not None:
            raise ValueError("model.calibration legacy mapping must be removed when model.calibration_registry governs forward probabilities")
    else:
        calibration = model_cfg.get("calibration")
        if calibration is not None:
            if not isinstance(calibration, dict):
                raise ValueError("model.calibration must be a mapping")
            for key in ["artifact_name", "artifact_version", "bins", "mapped"]:
                if key not in calibration:
                    raise KeyError(f"Missing model.calibration.{key}")
            bins = calibration.get("bins") or []
            mapped = calibration.get("mapped") or []
            if len(bins) != len(mapped) or len(bins) < 2:
                raise ValueError("model.calibration bins/mapped must be same length and contain at least 2 points")


def _validate_config(cfg: Dict[str, Any]) -> None:
    normalized = normalize_runtime_config(cfg)
    cfg.clear()
    cfg.update(normalized)

    req = ["ingestion", "storage", "system", "network", "validation"]
    for s in req:
        if s not in cfg:
            raise KeyError(f"Config missing section: {s}")

    unsupported = find_unsupported_config_keys(cfg)
    if unsupported:
        first_path = sorted(unsupported.keys())[0]
        raise ValueError(f"Unsupported config key '{first_path}': {unsupported[first_path]}")

    for k in ["duckdb_path", "cycle_lock_path", "writer_lock_path"]:
        if k not in cfg["storage"]:
            raise KeyError(f"Missing storage.{k}")

    if "watchlist" not in cfg["ingestion"]:
        raise KeyError("Missing ingestion.watchlist")
    if "cadence_minutes" not in cfg["ingestion"]:
        raise KeyError("Missing ingestion.cadence_minutes")

    val_cfg = cfg.get("validation", {})
    if "horizons_minutes" not in val_cfg:
        raise KeyError("Missing validation.horizons_minutes")

    if "alignment_tolerance_sec" not in val_cfg:
        raise KeyError("Missing validation.alignment_tolerance_sec")
    if "emit_to_close_horizon" not in val_cfg:
        raise KeyError("Missing validation.emit_to_close_horizon")
    if "use_default_required_features" not in val_cfg:
        raise KeyError("Missing validation.use_default_required_features")

    for key in ["invalid_after_minutes", "fallback_max_age_minutes", "tolerance_minutes", "max_horizon_drift_minutes"]:
        if key not in val_cfg:
            raise KeyError(f"Missing validation.{key}")
        if type(val_cfg[key]) is not int or val_cfg[key] <= 0:
            raise ValueError(f"validation.{key} must be a positive integer")

    if "flat_threshold_pct" not in val_cfg:
        raise KeyError("Missing validation.flat_threshold_pct")
    if not isinstance(val_cfg["flat_threshold_pct"], (float, int)) or val_cfg["flat_threshold_pct"] < 0:
        raise ValueError("validation.flat_threshold_pct must be a positive float")

    if "horizon_weights_source" not in val_cfg:
        raise KeyError("Missing validation.horizon_weights_source")

    src = val_cfg["horizon_weights_source"]
    if src not in ("model", "explicit"):
        raise ValueError(f"validation.horizon_weights_source must be 'model' or 'explicit', got '{src}'")

    horizons = [str(h) for h in val_cfg["horizons_minutes"]]
    if val_cfg["emit_to_close_horizon"]:
        horizons.append("to_close")

    if "horizon_critical_features" not in val_cfg:
        raise KeyError("Missing validation.horizon_critical_features")

    for h in horizons:
        if h not in val_cfg["horizon_critical_features"]:
            raise KeyError(f"Missing validation.horizon_critical_features for horizon '{h}'")

    if src == "model":
        if "model" not in cfg or "weights" not in cfg["model"] or not cfg["model"]["weights"]:
            raise KeyError("Missing or empty model.weights when horizon_weights_source is 'model'")
        if "horizon_weights_overrides" not in val_cfg:
            raise KeyError("Missing validation.horizon_weights_overrides")
        for h in horizons:
            if h not in val_cfg["horizon_weights_overrides"]:
                raise KeyError(f"Missing validation.horizon_weights_overrides for horizon '{h}'")
        inactive_explicit = val_cfg.get("horizon_weights") or {}
        if isinstance(inactive_explicit, dict) and any(bool(v) for v in inactive_explicit.values()):
            raise ValueError("validation.horizon_weights must be empty when validation.horizon_weights_source='model'")
    elif src == "explicit":
        if "horizon_weights" not in val_cfg:
            raise KeyError("Missing validation.horizon_weights")
        for h in horizons:
            if h not in val_cfg["horizon_weights"]:
                raise KeyError(f"Missing validation.horizon_weights for horizon '{h}'")
        inactive_overrides = val_cfg.get("horizon_weights_overrides") or {}
        if isinstance(inactive_overrides, dict) and any(bool(v) for v in inactive_overrides.values()):
            raise ValueError("validation.horizon_weights_overrides must be empty when validation.horizon_weights_source='explicit'")

    label_contract = val_cfg.get("label_contract")
    if label_contract is not None:
        if not isinstance(label_contract, dict):
            raise ValueError("validation.label_contract must be a mapping")
        for key in ["label_version", "session_boundary_rule", "flat_threshold_policy", "threshold_policy_version"]:
            if key not in label_contract or not str(label_contract[key]).strip():
                raise KeyError(f"Missing validation.label_contract.{key}")

    model_cfg = cfg.get("model", {})
    if model_cfg:
        if not isinstance(model_cfg, dict):
            raise ValueError("model section must be a mapping")
        target_spec = model_cfg.get("target_spec")
        if target_spec is not None:
            if not isinstance(target_spec, dict):
                raise ValueError("model.target_spec must be a mapping")
            for key in ["target_name", "target_version"]:
                if key not in target_spec or not str(target_spec[key]).strip():
                    raise KeyError(f"Missing model.target_spec.{key}")

    _validate_governance_contract(cfg, horizons)
    validate_governance_policy_config(cfg)
    validate_adapt_config(cfg)

    unknown_keys = set()

    if src == "model":
        unknown_keys.update(k for k in cfg["model"]["weights"].keys() if k not in KNOWN_FEATURE_KEYS)
        for h in horizons:
            if val_cfg["horizon_weights_overrides"].get(h):
                unknown_keys.update(k for k in val_cfg["horizon_weights_overrides"][h].keys() if k not in KNOWN_FEATURE_KEYS)
    elif src == "explicit":
        for h in horizons:
            if val_cfg["horizon_weights"].get(h):
                unknown_keys.update(k for k in val_cfg["horizon_weights"][h].keys() if k not in KNOWN_FEATURE_KEYS)

    for h in horizons:
        crit_list = val_cfg["horizon_critical_features"].get(h, [])
        unknown_keys.update(k for k in crit_list if k not in KNOWN_FEATURE_KEYS)

    if unknown_keys:
        raise ValueError(f"Unknown feature keys in config: {sorted(list(unknown_keys))}")

    logger.info(
        "Runtime config contract validated",
        extra={"json": {"effective_runtime_config": summarize_effective_runtime_config(cfg)}},
    )


def build_plan(cfg: Dict[str, Any], plan_yaml: Dict[str, Any]) -> Tuple[List[PlannedCall], List[PlannedCall]]:
    def _parse(entries, *, section_name: str, market: bool = False) -> List[PlannedCall]:
        planned: List[PlannedCall] = []
        for idx, raw_entry in enumerate(entries or []):
            entry = resolve_endpoint_purpose_contract(
                raw_entry,
                section_name=section_name,
                entry_index=idx,
                require_explicit=False,
            )
            if entry["purpose"] == "disabled":
                continue
            planned.append(
                PlannedCall(
                    entry["name"],
                    entry["method"],
                    entry["path"],
                    entry["path_params"],
                    entry["query_params"],
                    market,
                    entry["purpose"],
                    entry["decision_path"],
                    entry["missing_affects_confidence"],
                    entry["stale_affects_confidence"],
                    entry["purpose_contract_version"],
                )
            )
        return planned

    core = _parse(plan_yaml.get("plans", {}).get("default", []), section_name="default")
    market = []
    if cfg["ingestion"].get("enable_market_context"):
        market = _parse(plan_yaml.get("plans", {}).get("market_context", []), section_name="market_context", market=True)
    return core, market



def summarize_effective_endpoint_plan(cfg: Dict[str, Any], plan_yaml: Dict[str, Any]) -> Dict[str, Any]:
    plans = plan_yaml.get("plans", {}) or {}
    market_enabled = bool(cfg.get("ingestion", {}).get("enable_market_context"))

    def _entries(section: str, market: bool) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for idx, raw_entry in enumerate(plans.get(section, []) or []):
            entry = resolve_endpoint_purpose_contract(
                raw_entry,
                section_name=section,
                entry_index=idx,
                require_explicit=False,
            )
            items.append({
                "name": entry["name"],
                "method": entry["method"],
                "path": entry["path"],
                "purpose": entry["purpose"],
                "decision_path": entry["decision_path"],
                "missing_affects_confidence": entry["missing_affects_confidence"],
                "stale_affects_confidence": entry["stale_affects_confidence"],
                "purpose_contract_version": entry["purpose_contract_version"],
                "is_market": market,
            })
        return items

    default_entries = _entries("default", False)
    market_entries = _entries("market_context", True)
    fetched_market_entries = [x for x in market_entries if market_enabled and x["purpose"] != "disabled"]

    return {
        "market_context_enabled": market_enabled,
        "fetched_default": [x for x in default_entries if x["purpose"] != "disabled"],
        "disabled_default": [x for x in default_entries if x["purpose"] == "disabled"],
        "fetched_market_context": fetched_market_entries,
        "disabled_market_context": [x for x in market_entries if x["purpose"] == "disabled"],
    }

def _expand(call: PlannedCall, ticker: str, date_str: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    def _sub(v):
        if isinstance(v, str):
            return v.replace("{ticker}", ticker).replace("{date}", date_str)
        return v
    path_params = {k: _sub(v) for k, v in call.path_params.items()}
    if "{ticker}" in call.path:
        path_params["ticker"] = ticker
    query_params = {k: _sub(v) for k, v in call.query_params.items() if v is not None}
    return path_params, query_params

async def fetch_all(
    client: UwClient, 
    tickers: List[str], 
    date_str: str, 
    core: List[PlannedCall], 
    market: List[PlannedCall], 
    *, 
    max_concurrency: int
):
    tasks = []
    sem = asyncio.Semaphore(max(1, int(max_concurrency)))
    
    async def _one(tkr: str, c: PlannedCall, pp: Dict[str, Any], qp: Dict[str, Any]):
        async with sem:
            sig, res, cb = await client.request(c.method, c.path, path_params=pp, query_params=qp)
            return (tkr, c, sig, qp, res, cb)

    for tkr in sorted(tickers):
        for c in core:
            pp, qp = _expand(c, tkr, date_str)
            tasks.append(_one(tkr, c, pp, qp))
    for c in market:
        pp, qp = _expand(c, "", date_str)
        tasks.append(_one("__MARKET__", c, pp, qp))

    return await asyncio.gather(*tasks)

def _get_worst_freshness(states: List[FreshnessState]) -> FreshnessState:
    order = {
        FreshnessState.ERROR: 0, 
        FreshnessState.EMPTY_VALID: 1, 
        FreshnessState.STALE_CARRY: 2, 
        FreshnessState.FRESH: 3
    }
    if states:
        return min(states, key=lambda s: order[s])
    return FreshnessState.ERROR

def _is_valid_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and math.isfinite(v)

def _assessment_bucket(assessment: FeaturePolicyAssessment) -> str:
    if assessment.reason in (FeatureDecisionReason.MISSING_EFFECTIVE_TS, FeatureDecisionReason.INVALID_EFFECTIVE_TS):
        return "missing_ts"
    if assessment.reason == FeatureDecisionReason.FUTURE_TS_VIOLATION:
        return "future_ts"
    if assessment.reason == FeatureDecisionReason.JOIN_SKEW_VIOLATION:
        return "join_skew"
    if assessment.reason == FeatureDecisionReason.STALE_ENDPOINT_REJECTED:
        return "stale_rejected"
    if assessment.reason == FeatureDecisionReason.CARRY_FORWARD_SUPPRESSED:
        return "carry_forward_suppressed"
    if assessment.reason == FeatureDecisionReason.TIME_PROVENANCE_SUPPRESSED:
        return "provenance_suppressed"
    if assessment.reason == FeatureDecisionReason.BAD_FRESHNESS:
        return "bad_freshness"
    if assessment.reason == FeatureDecisionReason.MISSING_OR_INVALID_VALUE:
        return "invalid"
    return "accepted"


def _assessment_dq_reason(feature_key: str, assessment: Optional[FeaturePolicyAssessment]) -> str:
    if assessment is None:
        return f"{feature_key}_missing_or_invalid"
    if assessment.dq_reason_code:
        return assessment.dq_reason_code
    if assessment.reason == FeatureDecisionReason.OK:
        return f"{feature_key}_missing_or_invalid"
    return f"{feature_key}_{assessment.reason.value}"


def _log_feature_policy_assessment(assessment: FeaturePolicyAssessment) -> None:
    feature_key = assessment.feature_key
    shared = {
        "feature_key": feature_key,
        "reason": assessment.reason.value,
        "reason_detail": assessment.reason_detail,
        "policy": assessment.policy.name,
        "lag_class": assessment.policy.lag_class.value,
        "criticality": assessment.policy.criticality.value,
        "join_skew_tolerance_seconds": assessment.policy.join_skew_tolerance_seconds,
        "max_tolerated_age_seconds": assessment.policy.max_tolerated_age_seconds,
        "delta_sec": int(assessment.delta_seconds or 0) if assessment.delta_seconds is not None else None,
        "stale_age_seconds": assessment.stale_age_seconds,
        "policy_source": assessment.policy_source,
        "dq_reason_code": assessment.dq_reason_code,
    }

    if assessment.reason == FeatureDecisionReason.MISSING_EFFECTIVE_TS:
        structured_log(
            logger,
            logging.WARNING,
            event="missing_effective_timestamp",
            msg=f"feature_missing_effective_ts: {feature_key}",
            counter="feature_missing_effective_ts",
            **shared,
        )
    elif assessment.reason == FeatureDecisionReason.INVALID_EFFECTIVE_TS:
        structured_log(
            logger,
            logging.WARNING,
            event="invalid_effective_timestamp",
            msg=(f"feature_invalid_effective_ts (naive timezone): {feature_key}" if assessment.reason_detail == "naive_timezone" else f"feature_invalid_effective_ts (malformed): {feature_key}"),
            counter="feature_invalid_effective_ts",
            **shared,
        )
    elif assessment.reason == FeatureDecisionReason.FUTURE_TS_VIOLATION:
        structured_log(
            logger,
            logging.WARNING,
            event="future_timestamp_violation",
            msg=f"future_ts_violation: {feature_key} is ahead of asof_utc by {abs(int(assessment.delta_seconds or 0))}s",
            counter="future_ts_violation_count",
            **shared,
        )
    elif assessment.reason == FeatureDecisionReason.JOIN_SKEW_VIOLATION:
        structured_log(
            logger,
            logging.WARNING,
            event="join_skew_failure",
            msg=f"alignment_violation: {feature_key} misaligned by {int(assessment.delta_seconds or 0)}s",
            counter="join_skew_violation_count",
            **shared,
        )
    elif assessment.reason == FeatureDecisionReason.STALE_ENDPOINT_REJECTED:
        structured_log(
            logger,
            logging.WARNING,
            event="stale_endpoint",
            msg=f"stale_endpoint_rejected: {feature_key}",
            counter="stale_endpoint_rejection_count",
            **shared,
        )
    elif assessment.reason == FeatureDecisionReason.CARRY_FORWARD_SUPPRESSED:
        structured_log(
            logger,
            logging.WARNING,
            event="carry_forward_suppressed",
            msg=f"carry_forward_suppressed: {feature_key}",
            counter="carry_forward_suppression_count",
            **shared,
        )
    elif assessment.reason == FeatureDecisionReason.TIME_PROVENANCE_SUPPRESSED:
        structured_log(
            logger,
            logging.WARNING,
            event="time_provenance_suppressed",
            msg=f"time_provenance_suppressed: {feature_key}",
            counter="time_provenance_suppression_count",
            **shared,
        )
    elif assessment.reason == FeatureDecisionReason.MISSING_OR_INVALID_VALUE:
        structured_log(
            logger,
            logging.WARNING,
            event="invalid_feature",
            msg=f"feature missing or invalid value: {feature_key}",
            counter="invalid_feature_count",
            **shared,
        )



def _freshness_credit(assessment: Optional[FeaturePolicyAssessment]) -> float:
    if assessment is None or not assessment.include_in_scoring:
        return 0.0
    if assessment.freshness_state in ("FRESH", "EMPTY_VALID"):
        if assessment.degraded:
            return 0.75
        return 1.0
    if assessment.freshness_state == "STALE_CARRY":
        age = assessment.stale_age_seconds
        max_age = max(1, int(assessment.policy.max_tolerated_age_seconds))
        if age is None:
            return 0.5
        penalty_ratio = min(1.0, age / float(max_age))
        credit = max(0.1, 1.0 - penalty_ratio)
        if assessment.degraded:
            return min(credit, 0.9)
        return credit
    return 0.0



ZERO_WEIGHT_ABS_TOLERANCE = 1e-12


def _is_zero_weight(weight: Any) -> bool:
    try:
        return math.isclose(float(weight), 0.0, abs_tol=ZERO_WEIGHT_ABS_TOLERANCE)
    except (TypeError, ValueError):
        return True


def _extract_feature_use_contract(meta_json: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    meta = meta_json if isinstance(meta_json, dict) else {}
    raw_contract = meta.get("feature_use_contract") or {}
    if not isinstance(raw_contract, dict):
        raw_contract = {}

    source_endpoints = meta.get("source_endpoints") or []
    primary_source = source_endpoints[0] if source_endpoints and isinstance(source_endpoints[0], dict) else {}
    metric_lineage = meta.get("metric_lineage") or {}

    use_role = str(
        raw_contract.get("use_role")
        or meta.get("use_role")
        or primary_source.get("purpose")
        or metric_lineage.get("decision_path_role")
        or "signal-critical"
    )
    decision_path = raw_contract.get("decision_path")
    if decision_path is None:
        decision_path = primary_source.get("decision_path")
    if decision_path is None:
        decision_path = bool(use_role == "signal-critical")

    decision_eligible = raw_contract.get("decision_eligible")
    if decision_eligible is None:
        decision_eligible = meta.get("decision_eligible")
    if decision_eligible is None:
        decision_eligible = bool(use_role == "signal-critical" and decision_path)

    missing_affects_confidence = raw_contract.get("missing_affects_confidence")
    if missing_affects_confidence is None:
        missing_affects_confidence = meta.get("missing_affects_confidence")
    if missing_affects_confidence is None:
        missing_affects_confidence = bool(decision_eligible)

    stale_affects_confidence = raw_contract.get("stale_affects_confidence")
    if stale_affects_confidence is None:
        stale_affects_confidence = meta.get("stale_affects_confidence")
    if stale_affects_confidence is None:
        stale_affects_confidence = bool(decision_eligible)

    contract = {
        "contract_version": str(
            raw_contract.get("contract_version")
            or primary_source.get("purpose_contract_version")
            or metric_lineage.get("feature_use_contract_version")
            or "feature_use/v1"
        ),
        "use_role": use_role,
        "decision_path": bool(decision_path),
        "decision_eligible": bool(decision_eligible),
        "missing_affects_confidence": bool(missing_affects_confidence),
        "stale_affects_confidence": bool(stale_affects_confidence),
    }
    if not contract["decision_eligible"]:
        contract["missing_affects_confidence"] = False
        contract["stale_affects_confidence"] = False
    return contract


def _build_feature_use_contract_index(valid_features: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {
        str(feature.get("feature_key")): _extract_feature_use_contract(feature.get("meta_json"))
        for feature in valid_features
        if feature.get("feature_key") is not None
    }


def _resolve_horizon_decision_contract(
    *,
    raw_weights: Dict[str, float],
    required_features: List[str],
    feature_use_contracts: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    required_set = set(required_features)
    candidate_keys = sorted(set(raw_weights.keys()) | required_set)

    decision_weights: Dict[str, float] = {}
    decision_required: List[str] = []
    zero_weight_excluded: List[str] = []
    report_only_excluded: List[str] = []
    context_only_excluded: List[str] = []
    disabled_excluded: List[str] = []
    non_decision_eligible_excluded: List[str] = []
    explicit_critical_overrides: List[str] = []
    contract_violation_features: List[str] = []
    contract_snapshot: Dict[str, Dict[str, Any]] = {}

    for feature_key in candidate_keys:
        feature_contract = feature_use_contracts.get(feature_key)
        if feature_contract is not None:
            contract_snapshot[feature_key] = dict(feature_contract)

        has_weight = feature_key in raw_weights
        raw_weight = raw_weights.get(feature_key)
        has_nonzero_weight = has_weight and not _is_zero_weight(raw_weight)
        explicit_critical_override = feature_key in required_set

        if feature_contract is not None and not bool(feature_contract.get("decision_eligible")):
            use_role = str(feature_contract.get("use_role") or "non-decision")
            if use_role == "report-only":
                report_only_excluded.append(feature_key)
            elif use_role == "context-only":
                context_only_excluded.append(feature_key)
            elif use_role == "disabled":
                disabled_excluded.append(feature_key)
            else:
                non_decision_eligible_excluded.append(feature_key)

            if has_weight and not has_nonzero_weight and not explicit_critical_override:
                zero_weight_excluded.append(feature_key)
            if has_nonzero_weight or explicit_critical_override:
                contract_violation_features.append(feature_key)
            continue

        if has_weight and not has_nonzero_weight and not explicit_critical_override:
            zero_weight_excluded.append(feature_key)
            continue

        if has_nonzero_weight:
            decision_weights[feature_key] = float(raw_weight)
        if explicit_critical_override:
            decision_required.append(feature_key)
            if not has_nonzero_weight:
                explicit_critical_overrides.append(feature_key)

    target_features = sorted(set(decision_weights.keys()) | set(decision_required))

    return {
        "configured_weight_features": sorted(raw_weights.keys()),
        "configured_zero_weight_features": sorted([k for k, v in raw_weights.items() if _is_zero_weight(v)]),
        "resolved_weight_features": sorted(decision_weights.keys()),
        "resolved_critical_features": sorted(set(decision_required)),
        "resolved_target_features": target_features,
        "zero_weight_excluded_features": sorted(set(zero_weight_excluded)),
        "report_only_excluded_features": sorted(set(report_only_excluded)),
        "context_only_excluded_features": sorted(set(context_only_excluded)),
        "disabled_excluded_features": sorted(set(disabled_excluded)),
        "non_decision_eligible_excluded_features": sorted(set(non_decision_eligible_excluded)),
        "explicit_critical_override_features": sorted(set(explicit_critical_overrides)),
        "contract_violation_features": sorted(set(contract_violation_features)),
        "feature_contracts": contract_snapshot,
        "decision_weights": decision_weights,
    }


def _log_horizon_decision_path_exclusions(horizon: str, horizon_contract: Dict[str, Any]) -> None:
    exclusion_specs = (
        ("zero_weight_feature_excluded_count", "zero_weight_feature_excluded", logging.INFO, horizon_contract.get("zero_weight_excluded_features", [])),
        ("report_only_feature_excluded_count", "report_only_feature_excluded", logging.INFO, horizon_contract.get("report_only_excluded_features", [])),
        ("context_only_feature_excluded_count", "context_only_feature_excluded", logging.INFO, horizon_contract.get("context_only_excluded_features", [])),
        ("disabled_feature_excluded_count", "disabled_feature_excluded", logging.INFO, horizon_contract.get("disabled_excluded_features", [])),
        ("non_decision_feature_excluded_count", "non_decision_feature_excluded", logging.INFO, horizon_contract.get("non_decision_eligible_excluded_features", [])),
    )
    for counter, event, level, feature_keys in exclusion_specs:
        if feature_keys:
            structured_log(
                logger,
                level,
                event=event,
                msg=f"{event}: {', '.join(feature_keys)}",
                counter=counter,
                horizon=horizon,
                feature_keys=feature_keys,
            )

    contract_violation_features = horizon_contract.get("contract_violation_features", [])
    if contract_violation_features:
        structured_log(
            logger,
            logging.WARNING,
            event="decision_path_contract_violation",
            msg=f"decision_path_contract_violation: {', '.join(contract_violation_features)}",
            counter="decision_path_contract_violation_count",
            horizon=horizon,
            feature_keys=contract_violation_features,
        )



def _log_ood_assessment(horizon: str, assessment: OODAssessment) -> None:
    if assessment.state == OODState.IN_DISTRIBUTION:
        structured_log(
            logger,
            logging.INFO,
            event="ood_assessed",
            msg=f"ood_assessed: {assessment.primary_reason}",
            counter="ood_in_distribution_count",
            horizon=horizon,
            ood_state=assessment.state.value,
            primary_reason=assessment.primary_reason,
            coverage_ratio=assessment.coverage_ratio,
            decision_feature_count=assessment.total_decision_feature_count,
            valid_decision_feature_count=assessment.valid_decision_feature_count,
        )
        return

    counter = "ood_unknown_count"
    level = logging.WARNING
    if assessment.state == OODState.DEGRADED:
        counter = "ood_degraded_count"
        level = logging.INFO
    elif assessment.state == OODState.OUT_OF_DISTRIBUTION:
        counter = "ood_rejection_count"

    structured_log(
        logger,
        level,
        event="ood_assessed",
        msg=f"ood_assessed: {assessment.primary_reason}",
        counter=counter,
        horizon=horizon,
        ood_state=assessment.state.value,
        primary_reason=assessment.primary_reason,
        reasons=list(assessment.reasons),
        coverage_ratio=assessment.coverage_ratio,
        decision_feature_count=assessment.total_decision_feature_count,
        valid_decision_feature_count=assessment.valid_decision_feature_count,
        degraded_feature_keys=list(assessment.degraded_feature_keys),
        missing_feature_keys=list(assessment.missing_feature_keys),
        session_mismatch_features=list(assessment.session_mismatch_features),
        normalization_failure_features=list(assessment.normalization_failure_features),
        output_domain_missing_features=list(assessment.output_domain_missing_features),
    )

    if assessment.boundary_violation_features:
        structured_log(
            logger,
            logging.WARNING,
            event="ood_feature_boundary_violation",
            msg="ood_feature_boundary_violation",
            counter="ood_feature_boundary_violation_count",
            horizon=horizon,
            ood_state=assessment.state.value,
            boundary_violation_features=assessment.boundary_violation_features,
        )

    if assessment.output_domain_missing_features:
        structured_log(
            logger,
            logging.INFO,
            event="ood_output_domain_contract_missing",
            msg="ood_output_domain_contract_missing",
            counter="ood_output_domain_missing_count",
            horizon=horizon,
            ood_state=assessment.state.value,
            output_domain_missing_features=list(assessment.output_domain_missing_features),
            output_domain_contract_issues=assessment.output_domain_contract_issues,
        )


def _log_calibration_selection(horizon: str, selection: CalibrationSelectionResult) -> None:
    if selection.artifact is not None:
        structured_log(
            logger,
            logging.INFO,
            event="calibration_artifact_selected",
            msg=f"calibration_artifact_selected: {selection.artifact.artifact_version}",
            counter="calibration_artifact_selected_count",
            horizon=horizon,
            reason_code=selection.reason_code,
            reasons=list(selection.reasons),
            registry_version=selection.registry_version,
            registry_source=selection.registry_source,
            request=selection.request.to_dict(),
            calibration_scope=selection.artifact.calibration_scope,
            artifact_name=selection.artifact.artifact_name,
            artifact_version=selection.artifact.artifact_version,
            artifact_hash=selection.artifact.artifact_hash,
        )
        return

    counter = "calibration_artifact_missing_count"
    if selection.reason_code in {"TARGET_MISMATCH", "HORIZON_MISMATCH", "SESSION_MISMATCH", "REGIME_MISMATCH", "REPLAY_MODE_MISMATCH"}:
        counter = "calibration_scope_mismatch_count"

    structured_log(
        logger,
        logging.WARNING,
        event="calibration_artifact_unavailable",
        msg=f"calibration_artifact_unavailable: {selection.reason_code}",
        counter=counter,
        horizon=horizon,
        reason_code=selection.reason_code,
        reasons=list(selection.reasons),
        registry_version=selection.registry_version,
        registry_source=selection.registry_source,
        request=selection.request.to_dict(),
        invalid_entries=list(selection.invalid_entries),
    )

def generate_predictions(
    cfg: Dict[str, Any],
    snapshot_id: int,
    valid_features: List[Dict[str, Any]],
    asof_utc: dt.datetime,
    session_enum: SessionState,
    sec_to_close: Optional[float],
    endpoint_coverage: float,
    replay_mode: Any = ReplayMode.LIVE_LIKE_OBSERVED,
) -> List[Dict[str, Any]]:
    """
    Centralized Decision Window and Gating pipeline.
    Shared identically between live ingestion and replay engine to guarantee governance parity.
    """
    feature_value_map = {f["feature_key"]: f["feature_value"] for f in valid_features}
    feature_use_contracts_by_key = _build_feature_use_contract_index(valid_features)
    cadence_sec = int(cfg["ingestion"]["cadence_minutes"]) * 60
    effective_replay_mode = _normalize_prediction_replay_mode(replay_mode)

    ts_list: List[dt.datetime] = []
    alignment_violations: List[str] = []
    missing_ts_features: List[str] = []
    future_ts_features: List[str] = []
    normalized_future_ts_features: List[str] = []
    stale_rejected_features: List[str] = []
    carry_forward_suppressed_features: List[str] = []
    provenance_suppressed_features: List[str] = []
    invalid_value_features: List[str] = []
    policy_degraded_features: List[str] = []
    aligned_features: List[Dict[str, Any]] = []
    assessments_by_feature: Dict[str, FeaturePolicyAssessment] = {}

    for f in valid_features:
        assessment = assess_feature_freshness(
            f,
            asof_utc=asof_utc,
            cadence_seconds=cadence_sec,
            cfg=cfg,
        )
        assessments_by_feature[assessment.feature_key] = assessment

        if assessment.normalized_future_ts:
            logger.info(
                f"normalized_future_ts: {assessment.feature_key} timestamp clamped to {asof_utc.isoformat()}",
                extra={
                    "counter": "normalized_future_ts_count",
                    "feature_key": assessment.feature_key,
                    "drift_sec": 0 if assessment.delta_seconds is None else abs(int(assessment.delta_seconds)),
                },
            )
            meta = f.setdefault("meta_json", {})
            metric_lineage = meta.setdefault("metric_lineage", {})
            metric_lineage["effective_ts_utc"] = asof_utc.isoformat()
            details = meta.setdefault("details", {})
            details["clamped_future_ts"] = True
            normalized_future_ts_features.append(assessment.feature_key)

        if assessment.include_in_scoring:
            aligned_features.append(f)
            if assessment.effective_ts is not None:
                ts_list.append(assessment.effective_ts)
            if assessment.degraded:
                policy_degraded_features.append(assessment.feature_key)
            continue

        _log_feature_policy_assessment(assessment)
        bucket = _assessment_bucket(assessment)
        if bucket == "missing_ts":
            missing_ts_features.append(assessment.feature_key)
        elif bucket == "future_ts":
            future_ts_features.append(assessment.feature_key)
        elif bucket == "join_skew":
            alignment_violations.append(f"{assessment.feature_key}_delta_{int(assessment.delta_seconds or 0)}s")
        elif bucket == "stale_rejected":
            stale_rejected_features.append(assessment.feature_key)
        elif bucket == "carry_forward_suppressed":
            carry_forward_suppressed_features.append(assessment.feature_key)
        elif bucket == "provenance_suppressed":
            provenance_suppressed_features.append(assessment.feature_key)
        elif bucket == "invalid":
            invalid_value_features.append(assessment.feature_key)

    feat_dict = {f["feature_key"]: f["feature_value"] for f in aligned_features}

    source_ts_min = min(ts_list) if ts_list else None
    source_ts_max = max(ts_list) if ts_list else None
    is_aligned = not any(
        [
            alignment_violations,
            missing_ts_features,
            future_ts_features,
            stale_rejected_features,
            carry_forward_suppressed_features,
            provenance_suppressed_features,
        ]
    )

    base_gate = DecisionGate(
        data_quality_state=DataQualityState.VALID,
        risk_gate_status=RiskGateStatus.PASS,
        decision_state=SignalState.NEUTRAL,
    )

    if session_enum == SessionState.CLOSED:
        base_gate = base_gate.block("session_closed", invalid=True)

    if not feat_dict and valid_features:
        base_gate = base_gate.block("all_features_excluded_by_freshness_join_policy", invalid=True)

    def evaluate_horizon_gate(h_str: str, current_base_gate: DecisionGate) -> Tuple[DecisionGate, Dict[str, float], float, Dict[str, Any], List[str], Dict[str, Any]]:
        old_dq_state = current_base_gate.data_quality_state

        base_reqs = cfg["validation"]["horizon_critical_features"][h_str]

        if cfg["validation"]["horizon_weights_source"] == "model":
            raw_weights = dict(cfg["model"]["weights"])
            overrides = cfg["validation"]["horizon_weights_overrides"][h_str]
            if overrides:
                raw_weights.update(overrides)
        else:
            raw_weights = dict(cfg["validation"]["horizon_weights"][h_str])

        use_default_reqs = cfg["validation"]["use_default_required_features"]

        reqs = list(base_reqs)
        if use_default_reqs:
            session_default_criticals = {
                SessionState.RTH: ["spot", "net_gex_sign", "smart_whale_pressure", "oi_pressure"],
                SessionState.PREMARKET: ["spot", "dealer_vanna"],
                SessionState.AFTERHOURS: ["spot"],
                SessionState.CLOSED: ["spot"],
            }.get(session_enum, ["spot"])
            reqs = list(set(reqs) | set(session_default_criticals))
        elif not reqs:
            logger.info(f"No explicit critical features for horizon {h_str} and defaults disabled. Empty critical set allowed.")

        horizon_contract = _resolve_horizon_decision_contract(
            raw_weights=raw_weights,
            required_features=reqs,
            feature_use_contracts=feature_use_contracts_by_key,
        )
        horizon_contract["use_default_required_features"] = use_default_reqs
        _log_horizon_decision_path_exclusions(h_str, horizon_contract)

        weights = dict(horizon_contract.pop("decision_weights"))
        reqs = list(horizon_contract["resolved_critical_features"])
        target_features = set(horizon_contract["resolved_target_features"])

        if not target_features:
            configured_targets = sorted(set(raw_weights.keys()) | set(base_reqs))
            invalid_contract_reason = "invalid_contract_no_decision_targets" if configured_targets else "invalid_contract_no_targets"
            logger.error(
                f"invalid_contract_horizon_{h_str}: No decision-eligible target features defined.",
                extra={
                    "counter": "invalid_horizon_contract",
                    "horizon": h_str,
                    "configured_targets": configured_targets,
                    "decision_path_exclusions": {
                        "zero_weight": horizon_contract.get("zero_weight_excluded_features", []),
                        "report_only": horizon_contract.get("report_only_excluded_features", []),
                        "context_only": horizon_contract.get("context_only_excluded_features", []),
                        "contract_violations": horizon_contract.get("contract_violation_features", []),
                    },
                },
            )
            return (
                current_base_gate.block(f"{invalid_contract_reason}_{h_str}", invalid=True),
                weights,
                0.0,
                horizon_contract,
                [invalid_contract_reason],
                {
                    "configured_targets": configured_targets,
                    "decision_path_exclusions": {
                        "zero_weight": horizon_contract.get("zero_weight_excluded_features", []),
                        "report_only": horizon_contract.get("report_only_excluded_features", []),
                        "context_only": horizon_contract.get("context_only_excluded_features", []),
                        "contract_violations": horizon_contract.get("contract_violation_features", []),
                    },
                },
            )

        logger.info(
            f"resolved_horizon_{h_str}_features",
            extra={
                "counter": "resolved_horizon_features",
                "horizon": h_str,
                "critical_features": reqs,
                "weighted_features": sorted(weights.keys()),
                "target_features": sorted(target_features),
                "decision_path_exclusions": {
                    "zero_weight": horizon_contract.get("zero_weight_excluded_features", []),
                    "report_only": horizon_contract.get("report_only_excluded_features", []),
                    "context_only": horizon_contract.get("context_only_excluded_features", []),
                },
            },
        )

        valid_target_count = 0.0
        dq_reasons: List[str] = []

        for k in target_features:
            assessment = assessments_by_feature.get(k)
            if assessment is None:
                dq_reasons.append(f"{k}_missing_or_invalid")
                continue
            if not assessment.include_in_scoring:
                dq_reasons.append(_assessment_dq_reason(k, assessment))
                continue

            valid_target_count += _freshness_credit(assessment)
            if assessment.dq_reason_code:
                dq_reasons.append(assessment.dq_reason_code)

        decision_dq = valid_target_count / len(target_features) if target_features else 0.0

        if dq_reasons:
            logger.info(
                f"dq_reasons_horizon_{h_str}",
                extra={"counter": "decision_dq_reasons", "horizon": h_str, "reasons": dq_reasons, "decision_dq": decision_dq},
            )

        missing_criticals = [k for k in reqs if k not in feat_dict or feat_dict[k] is None or not math.isfinite(feat_dict[k])]
        missing_non_criticals = [
            k for k in weights.keys()
            if k not in reqs and (k not in feat_dict or feat_dict[k] is None or not math.isfinite(feat_dict[k]))
        ]

        valid_target_keys = {k for k in target_features if k in feat_dict and feat_dict[k] is not None and math.isfinite(feat_dict[k])}

        h_gate = current_base_gate
        target_diagnostics: Dict[str, Any] = {}

        if len(valid_target_keys) == 0:
            expected_keys = sorted(target_features)
            filter_reasons = {
                "missing": sorted([k for k in expected_keys if assessments_by_feature.get(k) is None]),
                "misaligned": sorted([k for k in expected_keys if assessments_by_feature.get(k) and assessments_by_feature[k].reason == FeatureDecisionReason.JOIN_SKEW_VIOLATION]),
                "future_ts": sorted([k for k in expected_keys if assessments_by_feature.get(k) and assessments_by_feature[k].reason == FeatureDecisionReason.FUTURE_TS_VIOLATION]),
                "missing_ts": sorted([k for k in expected_keys if assessments_by_feature.get(k) and assessments_by_feature[k].reason in (FeatureDecisionReason.MISSING_EFFECTIVE_TS, FeatureDecisionReason.INVALID_EFFECTIVE_TS)]),
                "stale": sorted([k for k in expected_keys if assessments_by_feature.get(k) and assessments_by_feature[k].reason in (FeatureDecisionReason.STALE_ENDPOINT_REJECTED, FeatureDecisionReason.CARRY_FORWARD_SUPPRESSED)]),
                "invalid": sorted([k for k in expected_keys if assessments_by_feature.get(k) and assessments_by_feature[k].reason == FeatureDecisionReason.MISSING_OR_INVALID_VALUE]),
            }

            target_diagnostics = {
                "expected_keys": expected_keys,
                "missing_keys": filter_reasons["missing"],
                "invalid_keys": filter_reasons["invalid"],
                "misaligned_keys": filter_reasons["misaligned"],
                "future_ts_keys": filter_reasons["future_ts"],
                "missing_ts_keys": filter_reasons["missing_ts"],
                "stale_keys": filter_reasons["stale"],
                "decision_path_exclusions": {
                    "zero_weight": horizon_contract.get("zero_weight_excluded_features", []),
                    "report_only": horizon_contract.get("report_only_excluded_features", []),
                    "context_only": horizon_contract.get("context_only_excluded_features", []),
                    "contract_violations": horizon_contract.get("contract_violation_features", []),
                },
            }

            actual_missing_criticals = sorted([k for k in reqs if k not in feat_dict or feat_dict[k] is None or not math.isfinite(feat_dict[k])])
            excluded_align_count = len(filter_reasons["misaligned"]) + len(filter_reasons["future_ts"]) + len(filter_reasons["missing_ts"]) + len(filter_reasons["stale"])

            filter_str = json.dumps(filter_reasons, sort_keys=True).replace('"', "'")
            block_reason = f"no_horizon_target_features_after_alignment | expected: {expected_keys} | filtered: {filter_str}"

            logger.warning(
                f"horizon_{h_str}_blocked_zero_targets",
                extra={
                    "counter": "horizon_blocked_zero_targets",
                    "horizon": h_str,
                    "blocked_reason": "no_horizon_target_features_after_alignment",
                    "expected_targets": expected_keys,
                    "missing_criticals": actual_missing_criticals,
                    "excluded_by_alignment_count": excluded_align_count,
                    "filter_reasons": filter_reasons,
                    "decision_path_exclusions": target_diagnostics["decision_path_exclusions"],
                },
            )
            h_gate = h_gate.block(block_reason, invalid=True, missing_features=actual_missing_criticals)

        if missing_criticals and h_gate.risk_gate_status != RiskGateStatus.BLOCKED:
            missing_criticals = sorted(missing_criticals)
            logger.warning(
                f"no_signal_due_to_critical_missing_horizon_{h_str}",
                extra={
                    "counter": "critical_endpoint_missing_count",
                    "missing_features": missing_criticals,
                    "horizon": h_str,
                },
            )
            reasons = []
            for mk in missing_criticals:
                reason_str = _assessment_dq_reason(mk, assessments_by_feature.get(mk))
                reasons.append(reason_str)
                logger.warning(
                    f"critical_feature_failed_{h_str}: {reason_str}",
                    extra={"counter": "critical_endpoint_missing_count", "feature_key": mk, "horizon": h_str, "reason": reason_str},
                )
            h_gate = h_gate.block(f"critical_features_failed_{h_str}: {','.join(reasons)}", invalid=True, missing_features=missing_criticals)
        elif missing_non_criticals and h_gate.risk_gate_status == RiskGateStatus.PASS:
            reasons = [_assessment_dq_reason(mk, assessments_by_feature.get(mk)) for mk in missing_non_criticals]
            h_gate = h_gate.degrade(f"non_critical_features_failed_{h_str}: {','.join(reasons)}", partial=True)

        if decision_dq < 0.5 and h_gate.risk_gate_status == RiskGateStatus.PASS:
            h_gate = h_gate.degrade(f"low_decision_dq_{decision_dq:.2f}", partial=True)
        elif decision_dq < 1.0 and h_gate.data_quality_state == DataQualityState.VALID:
            h_gate = replace(h_gate, data_quality_state=DataQualityState.PARTIAL, degraded_reasons=h_gate.degraded_reasons + (f"suboptimal_decision_dq_{decision_dq:.2f}",))

        if h_gate.risk_gate_status == RiskGateStatus.DEGRADED and h_gate.data_quality_state == DataQualityState.VALID:
            h_gate = replace(h_gate, data_quality_state=DataQualityState.PARTIAL, degraded_reasons=h_gate.degraded_reasons + ("risk_degraded_implies_partial_quality",))

        if h_gate.data_quality_state != old_dq_state:
            logger.info(
                f"gate_state_transition_horizon_{h_str}: {old_dq_state.value} -> {h_gate.data_quality_state.value}",
                extra={
                    "counter": "gate_state_transition",
                    "horizon": h_str,
                    "old_state": old_dq_state.value,
                    "new_state": h_gate.data_quality_state.value,
                    "risk_status": h_gate.risk_gate_status.value,
                },
            )

        return h_gate, weights, decision_dq, horizon_contract, sorted(set(dq_reasons)), target_diagnostics

    def _build_pred_meta(pred, horizon_contract, decision_dq, dq_reasons, target_diags, target_spec, label_contract, ood_assessment, calibration_selection):
        pred_meta = pred.meta
        pred_meta["endpoint_coverage"] = endpoint_coverage
        pred_meta["alignment_diagnostics"] = {
            "excluded_misaligned_count": len(alignment_violations),
            "excluded_missing_ts_count": len(missing_ts_features),
            "excluded_future_ts_count": len(future_ts_features),
            "normalized_future_ts_count": len(normalized_future_ts_features),
            "misaligned_keys": alignment_violations,
            "missing_ts_keys": missing_ts_features,
            "future_ts_keys": future_ts_features,
            "normalized_keys": normalized_future_ts_features,
        }
        pred_meta["freshness_registry_diagnostics"] = {
            "stale_endpoint_rejected_count": len(stale_rejected_features),
            "carry_forward_suppressed_count": len(carry_forward_suppressed_features),
            "time_provenance_suppressed_count": len(provenance_suppressed_features),
            "policy_degraded_count": len(policy_degraded_features),
            "invalid_value_count": len(invalid_value_features),
            "stale_endpoint_rejected_keys": stale_rejected_features,
            "carry_forward_suppressed_keys": carry_forward_suppressed_features,
            "time_provenance_suppressed_keys": provenance_suppressed_features,
            "policy_degraded_keys": policy_degraded_features,
            "invalid_value_keys": invalid_value_features,
            "feature_policies": {
                k: {
                    "policy": a.policy.name,
                    "lag_class": a.policy.lag_class.value,
                    "join_skew_tolerance_seconds": a.policy.join_skew_tolerance_seconds,
                    "max_tolerated_age_seconds": a.policy.max_tolerated_age_seconds,
                    "policy_source": a.policy_source,
                    "reason": a.reason.value,
                }
                for k, a in assessments_by_feature.items()
            },
        }
        pred_meta["prediction_contract"] = {
            "target_name": target_spec.target_name,
            "target_version": target_spec.target_version,
            "label_version": label_contract.label_version,
            "session_boundary_rule": label_contract.session_boundary_rule,
            "threshold_policy_version": label_contract.threshold_policy_version,
            "target_spec": target_spec.to_dict(),
            "label_contract": label_contract.to_dict(),
            "calibration_selection_reason": calibration_selection.reason_code,
            "calibration_version": calibration_selection.artifact.artifact_version if calibration_selection.artifact is not None else None,
            "calibration_scope": calibration_selection.artifact.calibration_scope if calibration_selection.artifact is not None else None,
            "calibration_request_replay_mode": calibration_selection.request.replay_mode,
            "prediction_replay_mode": pred.replay_mode.value,
        }
        pred_meta["horizon_contract"] = horizon_contract
        pred_meta["decision_dq"] = decision_dq
        pred_meta["dq_reason_codes"] = dq_reasons
        pred_meta["suppression_reason"] = pred.suppression_reason
        pred_meta["ood_state"] = pred.ood_state.value
        pred_meta["ood_reason"] = ood_assessment.primary_reason
        pred_meta["ood_assessment"] = ood_assessment.to_dict()
        pred_meta["calibration_selection"] = calibration_selection.to_dict()
        pred_meta["replay_mode"] = pred.replay_mode.value
        pred_meta["gating"] = {"target_diagnostics": target_diags} if target_diags else {}
        return pred_meta

    predictions = []
    model_cfg = cfg.get("model", {}) or {}
    ood_policy = resolve_ood_policy(cfg)

    for h in cfg["validation"]["horizons_minutes"]:
        h_str = str(h)
        h_gate, weights, decision_dq, horizon_contract, dq_reasons, target_diags = evaluate_horizon_gate(h_str, base_gate)
        target_spec = build_prediction_target_spec(
            model_cfg,
            horizon_kind="FIXED",
            horizon_minutes=int(h),
            flat_threshold_pct=cfg.get("validation", {}).get("flat_threshold_pct"),
        )
        label_contract = build_label_contract_spec(
            model_cfg,
            cfg.get("validation", {}),
            flat_threshold_pct=cfg.get("validation", {}).get("flat_threshold_pct"),
            session_boundary_rule="TRUNCATE_TO_SESSION_CLOSE",
        )
        calibration_regime = resolve_calibration_regime(model_cfg)
        calibration_selection = select_calibration_artifact(
            model_cfg,
            target_spec=target_spec,
            horizon_kind="FIXED",
            horizon_minutes=int(h),
            session_state=session_enum,
            regime=calibration_regime,
            replay_mode=effective_replay_mode,
        )
        _log_calibration_selection(h_str, calibration_selection)
        ood_assessment = assess_operational_ood(
            feature_rows=valid_features,
            decision_feature_keys=horizon_contract["resolved_target_features"],
            session_state=session_enum,
            assessments_by_feature=assessments_by_feature,
            gate=h_gate,
            policy=ood_policy,
        )
        _log_ood_assessment(h_str, ood_assessment)
        pred = bounded_additive_score(
            feat_dict,
            decision_dq,
            weights,
            gate=h_gate,
            confidence_cap=float(model_cfg.get("confidence_cap", 1.0)),
            min_confidence=float(model_cfg.get("min_confidence", 0.0)),
            neutral_threshold=float(model_cfg.get("neutral_threshold", 0.15)),
            direction_margin=float(model_cfg.get("direction_margin", 0.05)),
            min_flat_prob=float(model_cfg.get("min_flat_prob", 0.20)),
            max_flat_prob=float(model_cfg.get("max_flat_prob", 0.80)),
            flat_from_data_quality_scale=float(model_cfg.get("flat_from_data_quality_scale", 1.0)),
            model_name=str(model_cfg.get("model_name", "bounded_additive_score")),
            model_version=str(model_cfg.get("model_version", "UNSPECIFIED")),
            target_spec=target_spec,
            calibration_artifact_ref=calibration_selection.artifact,
            ood_state=ood_assessment.state,
            ood_reason=ood_assessment.primary_reason,
            ood_policy=model_cfg.get("ood_probability_policy") or model_cfg.get("ood_runtime_policy") or {},
            replay_mode=effective_replay_mode,
        )

        window_id_fixed = hashlib.sha256(f"{snapshot_id}_FIXED_{h}".encode()).hexdigest()[:16]

        pred_meta = _build_pred_meta(pred, horizon_contract, decision_dq, dq_reasons, target_diags, target_spec, label_contract, ood_assessment, calibration_selection)

        predictions.append({
            "snapshot_id": snapshot_id, "horizon_minutes": int(h), "horizon_kind": "FIXED",
            "horizon_seconds": None, "start_price": feat_dict.get("spot"),
            "bias": pred.bias, "confidence": pred.confidence,
            "prob_up": pred.prob_up, "prob_down": pred.prob_down, "prob_flat": pred.prob_flat,
            "model_name": pred.model_name, "model_version": pred.model_version, "model_hash": pred.model_hash,
            "is_mock": not pred.gate.validation_eligible, "meta_json": pred_meta,
            "decision_state": pred.gate.decision_state.value, "risk_gate_status": pred.gate.risk_gate_status.value,
            "confidence_state": getattr(pred, "confidence_state", ConfidenceState.UNKNOWN).value,
            "data_quality_state": pred.gate.data_quality_state.value, "blocked_reasons": list(pred.gate.blocked_reasons),
            "degraded_reasons": list(pred.gate.degraded_reasons), "validation_eligible": pred.gate.validation_eligible,
            "gate_json": asdict(pred.gate), "source_ts_min_utc": source_ts_min, "source_ts_max_utc": source_ts_max,
            "critical_missing_count": len(pred.gate.critical_features_missing), "alignment_status": "ALIGNED" if is_aligned else "MISALIGNED",
            "decision_window_id": window_id_fixed,
            "replay_mode": pred.replay_mode.value,
        })

    emit_to_close = cfg["validation"]["emit_to_close_horizon"]

    if emit_to_close and sec_to_close is not None and sec_to_close > 0:
        h_gate, weights, decision_dq, horizon_contract, dq_reasons, target_diags = evaluate_horizon_gate("to_close", base_gate)
        target_spec = build_prediction_target_spec(
            model_cfg,
            horizon_kind="TO_CLOSE",
            horizon_minutes=0,
            flat_threshold_pct=cfg.get("validation", {}).get("flat_threshold_pct"),
        )
        label_contract = build_label_contract_spec(
            model_cfg,
            cfg.get("validation", {}),
            flat_threshold_pct=cfg.get("validation", {}).get("flat_threshold_pct"),
            session_boundary_rule="TRUNCATE_TO_SESSION_CLOSE",
        )
        calibration_regime = resolve_calibration_regime(model_cfg)
        calibration_selection = select_calibration_artifact(
            model_cfg,
            target_spec=target_spec,
            horizon_kind="TO_CLOSE",
            horizon_minutes=0,
            session_state=session_enum,
            regime=calibration_regime,
            replay_mode=effective_replay_mode,
        )
        _log_calibration_selection("to_close", calibration_selection)
        ood_assessment = assess_operational_ood(
            feature_rows=valid_features,
            decision_feature_keys=horizon_contract["resolved_target_features"],
            session_state=session_enum,
            assessments_by_feature=assessments_by_feature,
            gate=h_gate,
            policy=ood_policy,
        )
        _log_ood_assessment("to_close", ood_assessment)
        pred = bounded_additive_score(
            feat_dict,
            decision_dq,
            weights,
            gate=h_gate,
            confidence_cap=float(model_cfg.get("confidence_cap", 1.0)),
            min_confidence=float(model_cfg.get("min_confidence", 0.0)),
            neutral_threshold=float(model_cfg.get("neutral_threshold", 0.15)),
            direction_margin=float(model_cfg.get("direction_margin", 0.05)),
            min_flat_prob=float(model_cfg.get("min_flat_prob", 0.20)),
            max_flat_prob=float(model_cfg.get("max_flat_prob", 0.80)),
            flat_from_data_quality_scale=float(model_cfg.get("flat_from_data_quality_scale", 1.0)),
            model_name=str(model_cfg.get("model_name", "bounded_additive_score")),
            model_version=str(model_cfg.get("model_version", "UNSPECIFIED")),
            target_spec=target_spec,
            calibration_artifact_ref=calibration_selection.artifact,
            ood_state=ood_assessment.state,
            ood_reason=ood_assessment.primary_reason,
            ood_policy=model_cfg.get("ood_probability_policy") or model_cfg.get("ood_runtime_policy") or {},
            replay_mode=effective_replay_mode,
        )

        window_id_close = hashlib.sha256(f"{snapshot_id}_TOCLOSE_{sec_to_close}".encode()).hexdigest()[:16]

        pred_meta = _build_pred_meta(pred, horizon_contract, decision_dq, dq_reasons, target_diags, target_spec, label_contract, ood_assessment, calibration_selection)

        predictions.append({
            "snapshot_id": snapshot_id, "horizon_minutes": 0, "horizon_kind": "TO_CLOSE",
            "horizon_seconds": int(sec_to_close), "start_price": feat_dict.get("spot"),
            "bias": pred.bias, "confidence": pred.confidence,
            "prob_up": pred.prob_up, "prob_down": pred.prob_down, "prob_flat": pred.prob_flat,
            "model_name": pred.model_name, "model_version": pred.model_version, "model_hash": pred.model_hash,
            "is_mock": not pred.gate.validation_eligible, "meta_json": pred_meta,
            "decision_state": pred.gate.decision_state.value, "risk_gate_status": pred.gate.risk_gate_status.value,
            "confidence_state": getattr(pred, "confidence_state", ConfidenceState.UNKNOWN).value,
            "data_quality_state": pred.gate.data_quality_state.value, "blocked_reasons": list(pred.gate.blocked_reasons),
            "degraded_reasons": list(pred.gate.degraded_reasons), "validation_eligible": pred.gate.validation_eligible,
            "gate_json": asdict(pred.gate), "source_ts_min_utc": source_ts_min, "source_ts_max_utc": source_ts_max,
            "critical_missing_count": len(pred.gate.critical_features_missing), "alignment_status": "ALIGNED" if is_aligned else "MISALIGNED",
            "decision_window_id": window_id_close,
            "replay_mode": pred.replay_mode.value,
        })

    return predictions

def _ingest_once_impl(cfg: Dict[str, Any], catalog_path: str, config_path: str) -> None:
    _validate_config(cfg)
    registry = load_api_catalog(catalog_path)
    plan_yaml = load_endpoint_plan("src/config/endpoint_plan.yaml")
    
    validate_plan_coverage(plan_yaml)
    
    core, market = build_plan(cfg, plan_yaml)
    logger.info(
        "Endpoint plan resolved",
        extra={"json": {"effective_endpoint_plan": summarize_effective_endpoint_plan(cfg, plan_yaml)}},
    )
    tickers = [t.upper() for t in cfg["ingestion"]["watchlist"]]

    val_cfg = cfg["validation"]
    fallback_max_age_seconds = val_cfg["fallback_max_age_minutes"] * 60
    invalid_after_seconds = val_cfg["invalid_after_minutes"] * 60

    now_et = dt.datetime.now(ET)
    asof_et = floor_to_interval(now_et, int(cfg["ingestion"]["cadence_minutes"]))
    asof_utc = asof_et.astimezone(UTC)

    hours = get_market_hours(asof_et.date(), cfg["ingestion"])
    
    if not hours.is_trading_day:
        logger.info("Market Closed", extra={"json": {"reason": hours.reason}})
        return
        
    if asof_et < hours.ingest_start_et or asof_et >= hours.ingest_end_et:
        logger.info("Outside ingest window", extra={"json": {"asof_et": asof_et.isoformat()}})
        return

    sess_str = hours.get_session_label(asof_et)

    try:
        session_enum = coerce_session_state(sess_str)
    except ValueError:
        logger.error(
            f"Session contract violation: Invalid session label '{sess_str}'", 
            extra={
                "counter": "session_contract_violation_count", 
                "raw_session_label": sess_str,
                "tickers": tickers,
                "asof_timestamp": asof_utc.isoformat(),
                "processing_mode": cfg.get("system", {}).get("mode", "live")
            }
        )
        return
        
    close_utc = hours.market_close_et.astimezone(UTC) if hours.market_close_et else None
    post_utc = hours.post_end_et.astimezone(UTC) if hours.post_end_et else None
    sec_to_close = hours.seconds_to_close(asof_et)

    async def _run_fetch():
        net = cfg.get("network", {})
        sys_cfg = cfg.get("system", {})
        cb = net.get("circuit_breaker", {})
        
        async with UwClient(
            registry=registry, 
            base_url=net.get("base_url") or sys_cfg.get("base_url") or "https://api.unusualwhales.com",
            api_key_env=sys_cfg.get("api_key_env", "UW_API_KEY"), 
            timeout_seconds=net.get("timeout_seconds", 10.0),
            max_retries=net.get("max_retries", 3), 
            backoff_seconds=net.get("backoff_seconds", [1.0]),
            max_concurrent_requests=net.get("max_concurrent_requests", 20), 
            rate_limit_per_second=net.get("rate_limit_per_second", 10),
            circuit_failure_threshold=cb.get("failure_threshold", net.get("circuit_failure_threshold", 5)),
            circuit_cool_down_seconds=cb.get("cool_down_seconds", net.get("circuit_cool_down_seconds", 60)),
            circuit_half_open_max_calls=cb.get("half_open_max_calls", net.get("circuit_half_open_max_calls", 3))
        ) as client:
            return await fetch_all(
                client, tickers, asof_et.date().isoformat(), core, market, max_concurrency=net.get("max_concurrency", 20)
            )

    fetch_results = asyncio.run(_run_fetch())
    fetch_results.sort(key=lambda x: x[4].requested_at_utc if x[4].requested_at_utc is not None else 0.0)

    db = DbWriter(cfg["storage"]["duckdb_path"], cfg["storage"]["writer_lock_path"])

    try:
        with FileLock(cfg["storage"]["cycle_lock_path"]):
            with db.writer() as con:
                db.ensure_schema(con)
                db.upsert_tickers(con, tickers)
                
                try:
                    with open(config_path, "r", encoding="utf-8") as f:
                        cfg_text = f.read()
                except FileNotFoundError:
                    cfg_text = "{}"

                cfg_ver = db.insert_config(con, cfg_text)

                run_notes = f"SESS={sess_str}"
                if hours.reason != "NORMAL":
                    run_notes += f"; {hours.reason}"
                    
                run_id = db.begin_run(
                    con, asof_utc, sess_str, hours.is_trading_day, 
                    hours.is_early_close, cfg_ver, registry.catalog_hash, notes=run_notes
                )

                events_by_ticker: Dict[str, List[Tuple[int, uuid.UUID, Any, PlannedCall, str, PayloadAssessment]]] = {t: [] for t in tickers}
                max_seen_ts = {}

                for (tkr, call, sig, qp, res, cb) in fetch_results:
                    endpoint_id = db.upsert_endpoint(con, call.method, call.path, qp, registry)
                    
                    ev_key = (tkr, endpoint_id)
                    if ev_key in max_seen_ts and res.requested_at_utc < max_seen_ts[ev_key]:
                        logger.warning(
                            f"Out of order packet dropped from state mutation: {ev_key}", 
                            extra={"counter": "out_of_order_drops", "ticker": tkr, "endpoint_id": endpoint_id}
                        )
                        db.insert_raw_event(
                            con, run_id, tkr, endpoint_id, res.requested_at_utc, res.received_at_utc,
                            res.status_code, res.latency_ms, res.payload_hash, res.payload_json,
                            True, "OutOfOrder", "Dropped from state mutation due to latency shift", cb,
                        )
                        continue
                        
                    max_seen_ts[ev_key] = max(max_seen_ts.get(ev_key, 0.0), res.requested_at_utc)
                    
                    prev_state = db.get_endpoint_state(con, tkr, endpoint_id)
                    prev_hash = prev_state.last_payload_hash if prev_state else None
                    
                    source_time_hints = infer_source_time_hints(
                        payload_json=getattr(res, "payload_json", None),
                        response_headers=getattr(res, "response_headers", None),
                        explicit_event_time_raw=getattr(res, "event_time_utc", None),
                        explicit_publish_time_raw=getattr(res, "source_publish_time_utc", None),
                        explicit_effective_time_raw=getattr(res, "effective_time_utc", None),
                        explicit_revision=getattr(res, "source_revision", None),
                    )

                    event_id = db.insert_raw_event(
                        con, run_id, tkr, endpoint_id, res.requested_at_utc, res.received_at_utc,
                        res.status_code, res.latency_ms, res.payload_hash, res.payload_json,
                        res.retry_count > 0, res.error_type, res.error_message, cb,
                        source_publish_time_utc=source_time_hints.source_publish_time_utc,
                        source_revision=source_time_hints.source_revision,
                    )

                    attempt_ts_utc = to_utc_dt(
                        res.received_at_utc, 
                        fallback=to_utc_dt(res.requested_at_utc, fallback=dt.datetime.now(UTC))
                    )
                    
                    assessment = classify_payload(res, prev_hash, call.method, call.path, sess_str)

                    is_success_class = (
                        assessment.payload_class in (EndpointPayloadClass.SUCCESS_HAS_DATA, EndpointPayloadClass.SUCCESS_STALE) or 
                        (assessment.payload_class == EndpointPayloadClass.SUCCESS_EMPTY_VALID and assessment.empty_policy.name == "EMPTY_IS_DATA")
                    )
                    is_changed = (assessment.changed is True)

                    db.upsert_endpoint_state(
                        con, tkr, endpoint_id, str(event_id), res, 
                        attempt_ts_utc, is_success_class, is_changed
                    )

                    resolved = resolve_effective_payload(
                        str(event_id), asof_utc, assessment, prev_state,
                        fallback_max_age_seconds=fallback_max_age_seconds,
                        invalid_after_seconds=invalid_after_seconds,
                        source_event_time_raw=source_time_hints.event_time_utc,
                        source_publish_time_raw=source_time_hints.source_publish_time_utc,
                        effective_time_raw=source_time_hints.effective_time_utc,
                        received_at_raw=getattr(res, "received_at_utc", None),
                        processed_at_raw=attempt_ts_utc,
                        as_of_time_raw=asof_utc,
                        source_revision=source_time_hints.source_revision,
                    )
                    
                    enforced_freshness = resolved.freshness_state
                    enforced_reason = resolved.na_reason

                    if enforced_freshness == FreshnessState.STALE_CARRY and not resolved.used_event_id:
                        enforced_freshness = FreshnessState.ERROR
                        enforced_reason = NaReasonCode.NO_PRIOR_SUCCESS.value

                    if enforced_reason and NaReasonCode.STALE_TOO_OLD.value in enforced_reason:
                        enforced_freshness = FreshnessState.ERROR
                        logger.warning(
                            f"Stale packet dropped (too old): {ev_key}", 
                            extra={"counter": "stale_packets_dropped", "ticker": tkr, "endpoint_id": endpoint_id}
                        )

                    if assessment.error_reason and not enforced_reason:
                        enforced_reason = assessment.error_reason

                    resolved = replace(resolved, freshness_state=enforced_freshness, na_reason=enforced_reason)

                    if tkr not in events_by_ticker:
                        events_by_ticker[tkr] = []
                        
                    events_by_ticker[tkr].append((endpoint_id, event_id, resolved, call, sig, assessment))

                for tkr in tickers:
                    evs = events_by_ticker.get(tkr, [])
                    
                    valid_count = sum(1 for _, _, res, _, _, asmnt in evs if res.freshness_state == FreshnessState.FRESH or (res.freshness_state == FreshnessState.STALE_CARRY and res.used_event_id is not None) or (res.freshness_state == FreshnessState.EMPTY_VALID and asmnt.empty_policy.name == "EMPTY_IS_DATA"))
                    
                    endpoint_coverage = (valid_count / len(evs)) if evs else 0.0
                    logger.info("Snapshot endpoint coverage", extra={"counter": "endpoint_coverage_ratio", "ratio": endpoint_coverage, "ticker": tkr})

                    snapshot_id = db.insert_snapshot(
                        con, run_id=run_id, asof_ts_utc=asof_utc, ticker=tkr, session_label=sess_str,
                        is_trading_day=True, is_early_close=hours.is_early_close, data_quality_score=endpoint_coverage,
                        market_close_utc=close_utc, post_end_utc=post_utc, seconds_to_close=sec_to_close,
                    )
                    
                    active_used_ids = [
                        str(res.used_event_id) for _, _, res, _, _, _ in evs 
                        if res.used_event_id and res.freshness_state in (FreshnessState.FRESH, FreshnessState.STALE_CARRY, FreshnessState.EMPTY_VALID)
                    ]
                    
                    payloads_from_db = db.get_payloads_by_event_ids(con, active_used_ids)

                    effective_payloads: Dict[int, Any] = {}
                    contexts: Dict[int, EndpointContext] = {}
                    
                    for endpoint_id, event_id, res, call, sig, asmnt in evs:
                        op_id = registry.get(call.method, call.path).operation_id if registry.has(call.method, call.path) else None
                        
                        f_state = res.freshness_state
                        n_reason = res.na_reason
                        eff_payload = None

                        if res.used_event_id and f_state in (FreshnessState.FRESH, FreshnessState.STALE_CARRY, FreshnessState.EMPTY_VALID):
                            eff_payload = payloads_from_db.get(str(res.used_event_id))
                            if eff_payload is None:
                                f_state = FreshnessState.ERROR
                                n_reason = NaReasonCode.USED_EVENT_NOT_FOUND.value if str(res.used_event_id) not in payloads_from_db else NaReasonCode.PAYLOAD_JSON_INVALID.value

                        effective_payloads[endpoint_id] = eff_payload
                        
                        delta_sec = res.stale_age_seconds if res.stale_age_seconds is not None else 0
                        ep_asof = asof_utc - dt.timedelta(seconds=delta_sec)

                        ctx = EndpointContext(
                            endpoint_id=endpoint_id,
                            method=call.method,
                            path=call.path,
                            operation_id=op_id,
                            signature=sig,
                            used_event_id=res.used_event_id,
                            payload_class=res.payload_class.name,
                            freshness_state=f_state.value,
                            stale_age_min=(res.stale_age_seconds // 60) if res.stale_age_seconds is not None else None,
                            na_reason=n_reason,
                            endpoint_asof_ts_utc=ep_asof,
                            alignment_delta_sec=delta_sec,
                            effective_ts_utc=res.effective_ts_utc,
                            event_time_utc=res.event_time_utc,
                            source_publish_time_utc=res.source_publish_time_utc,
                            received_at_utc=res.received_at_utc,
                            processed_at_utc=res.processed_at_utc,
                            as_of_time_utc=res.as_of_time_utc,
                            source_revision=res.source_revision,
                            effective_time_source=res.effective_time_source,
                            timestamp_quality=res.timestamp_quality,
                            lagged=res.lagged,
                            time_provenance_degraded=res.time_provenance_degraded,
                            endpoint_name=call.name,
                            endpoint_purpose=call.purpose,
                            decision_path=call.decision_path,
                            missing_affects_confidence=call.missing_affects_confidence,
                            stale_affects_confidence=call.stale_affects_confidence,
                            purpose_contract_version=call.purpose_contract_version,
                        )
                        contexts[endpoint_id] = ctx
                            
                        src_meta = {
                            "method": call.method,
                            "path": call.path,
                            "operation_id": op_id,
                            "endpoint_id": endpoint_id,
                            "signature": sig,
                            "used_event_id": res.used_event_id,
                            "missing_keys": asmnt.missing_keys,
                            "purpose": call.purpose,
                            "decision_path": call.decision_path,
                            "missing_affects_confidence": call.missing_affects_confidence,
                            "stale_affects_confidence": call.stale_affects_confidence,
                            "purpose_contract_version": call.purpose_contract_version,
                        }

                        # Ticket 3: Persist strict validation keys for Replay Parity constraint
                        lineage_meta = MetaContract(
                            source_endpoints=[src_meta],
                            freshness_state=f_state.name,
                            stale_age_min=ctx.stale_age_min,
                            na_reason=n_reason,
                            details={
                                "effective_ts_utc": ctx.effective_ts_utc.isoformat() if ctx.effective_ts_utc else None,
                                "event_time_utc": ctx.event_time_utc.isoformat() if ctx.event_time_utc else None,
                                "source_publish_time_utc": ctx.source_publish_time_utc.isoformat() if ctx.source_publish_time_utc else None,
                                "received_at_utc": ctx.received_at_utc.isoformat() if ctx.received_at_utc else None,
                                "processed_at_utc": ctx.processed_at_utc.isoformat() if ctx.processed_at_utc else None,
                                "as_of_time_utc": ctx.as_of_time_utc.isoformat() if ctx.as_of_time_utc else None,
                                "endpoint_asof_ts_utc": ctx.endpoint_asof_ts_utc.isoformat() if ctx.endpoint_asof_ts_utc else None,
                                "alignment_delta_sec": ctx.alignment_delta_sec,
                                "truth_status": res.payload_class.name if hasattr(res.payload_class, "name") else str(res.payload_class),
                                "stale_age_seconds": res.stale_age_seconds,
                                "effective_time_source": ctx.effective_time_source,
                                "timestamp_quality": ctx.timestamp_quality,
                                "lagged": ctx.lagged,
                                "time_provenance_degraded": ctx.time_provenance_degraded,
                                "source_revision": ctx.source_revision,
                            }
                        )

                        db.insert_lineage(
                            con, snapshot_id=snapshot_id, endpoint_id=endpoint_id, used_event_id=res.used_event_id,
                            freshness_state=f_state.name, data_age_seconds=res.stale_age_seconds,
                            payload_class=res.payload_class.name, na_reason=n_reason, meta_json=asdict(lineage_meta)
                        )

                    features_insert_list, levels_insert_list = extract_all(effective_payloads, contexts)
                    
                    valid_features = []
                    valid_levels = []
                    malformed_count = 0
                    seen_keys = set()
                    
                    for f in features_insert_list:
                        if isinstance(f, dict) and "feature_key" in f and "meta_json" in f:
                            f_key = f["feature_key"]
                            f_val = f.get("feature_value")
                            
                            if f_val is not None and not _is_valid_num(f_val):
                                structured_log(
                                    logger,
                                    logging.WARNING,
                                    event="invalid_feature",
                                    msg="malformed feature row dropped (non-finite value)",
                                    counter="invalid_feature_count",
                                    row_type="feature",
                                    feature_key=f_key,
                                    row=f,
                                    reason="non_finite_value",
                                )
                                malformed_count += 1
                                continue
                                
                            if f_key in seen_keys:
                                raise RuntimeError(f"Duplicate feature key detected in insert list: {f_key}")
                            seen_keys.add(f_key)
                                
                            meta = f["meta_json"]
                            if isinstance(meta, dict) and all(k in meta for k in ["source_endpoints", "freshness_state", "stale_age_min", "na_reason", "details", "metric_lineage"]):
                                valid_features.append(f) 
                                continue
                        
                        structured_log(
                            logger,
                            logging.WARNING,
                            event="invalid_feature",
                            msg="malformed feature row dropped",
                            counter="invalid_feature_count",
                            row_type="feature",
                            row=f,
                            reason="contract_shape_invalid",
                        )
                        malformed_count += 1

                    for l in levels_insert_list:
                        if isinstance(l, dict) and "level_type" in l and "meta_json" in l:
                            p = l.get("price")
                            m = l.get("magnitude")
                            
                            if (p is not None and not _is_valid_num(p)) or (m is not None and not _is_valid_num(m)):
                                structured_log(
                                    logger,
                                    logging.WARNING,
                                    event="invalid_feature",
                                    msg="malformed derived level dropped (non-finite value)",
                                    counter="invalid_feature_count",
                                    row_type="derived_level",
                                    level_type=l.get("level_type"),
                                    row=l,
                                    reason="non_finite_value",
                                )
                                malformed_count += 1
                                continue
                                
                            meta = l["meta_json"]
                            if isinstance(meta, dict) and all(k in meta for k in ["source_endpoints", "freshness_state", "stale_age_min", "na_reason", "details", "metric_lineage"]):
                                valid_levels.append(l) 
                                continue
                        
                        structured_log(
                            logger,
                            logging.WARNING,
                            event="invalid_feature",
                            msg="malformed derived level dropped",
                            counter="invalid_feature_count",
                            row_type="derived_level",
                            row=l,
                            reason="contract_shape_invalid",
                        )
                        malformed_count += 1

                    total_outputs = len(features_insert_list) + len(levels_insert_list)
                    if total_outputs > 0 and (malformed_count / total_outputs) > 0.2:
                        logger.error(f"Extraction failed: {malformed_count}/{total_outputs} rows malformed (>20% threshold). Rollback enforced.")
                        raise RuntimeError(f"Extraction failed: {malformed_count}/{total_outputs} rows malformed.")
                    
                    # Task 7: Execute generate_predictions FIRST. It strictly evaluates and mutates 
                    # valid_features (clamping timestamps natively into meta_json).
                    predictions = generate_predictions(
                        cfg=cfg,
                        snapshot_id=snapshot_id,
                        valid_features=valid_features,
                        asof_utc=asof_utc,
                        session_enum=session_enum,
                        sec_to_close=sec_to_close,
                        endpoint_coverage=endpoint_coverage,
                        replay_mode=ReplayMode.LIVE_LIKE_OBSERVED,
                    )
                    
                    # Task 7: Persist the features *after* prediction gating mutations.
                    # This guarantees the stored JSON strictly matches what the gate utilized.
                    db.insert_features(con, snapshot_id, valid_features)
                    db.insert_levels(con, snapshot_id, valid_levels)
                    
                    for p in predictions:
                        prediction_id = db.insert_prediction(con, p)
                        if isinstance(p, dict):
                            p.setdefault("prediction_id", prediction_id)
                        log_prediction_decision(logger, p, ticker=tkr, asof_ts_utc=asof_utc)

                db.end_run(con, run_id)

    except FileLockError:
        logger.warning("Skipping cycle: lock held")


class IngestionEngine:
    def __init__(self, *, cfg: Dict[str, Any], catalog_path: str, config_path: str = "src/config/config.yaml"):
        self.cfg = cfg
        self.catalog_path = catalog_path
        self.config_path = config_path
        
    def run_cycle(self) -> None:
        _ingest_once_impl(self.cfg, self.catalog_path, self.config_path)