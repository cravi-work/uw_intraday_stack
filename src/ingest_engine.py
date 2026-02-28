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
from .config_loader import load_endpoint_plan
from .file_lock import FileLock, FileLockError
from .scheduler import ET, UTC, floor_to_interval, get_market_hours
from .storage import DbWriter
from .uw_client import UwClient
from .endpoint_rules import EmptyPayloadPolicy, validate_plan_coverage
from .features import extract_all
from .models import bounded_additive_score, Prediction, DecisionGate, DataQualityState, RiskGateStatus, SignalState, SessionState, ConfidenceState, KNOWN_FEATURE_KEYS
from .endpoint_truth import (
    EndpointContext,
    EndpointPayloadClass, 
    FreshnessState, 
    MetaContract,
    NaReasonCode,
    PayloadAssessment,
    classify_payload, 
    resolve_effective_payload, 
    to_utc_dt
)

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class PlannedCall:
    name: str
    method: str
    path: str
    path_params: Dict[str, Any]
    query_params: Dict[str, Any]
    is_market: bool

def _validate_config(cfg: Dict[str, Any]) -> None:
    req = ["ingestion", "storage", "system", "network", "validation"]
    for s in req:
        if s not in cfg:
            raise KeyError(f"Config missing section: {s}")
            
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
        
    # Task 6 & 12: Strict checks for hidden defaults (Including validator thresholds)
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
    elif src == "explicit":
        if "horizon_weights" not in val_cfg:
            raise KeyError("Missing validation.horizon_weights")
        for h in horizons:
            if h not in val_cfg["horizon_weights"]:
                raise KeyError(f"Missing validation.horizon_weights for horizon '{h}'")

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

def build_plan(cfg: Dict[str, Any], plan_yaml: Dict[str, Any]) -> Tuple[List[PlannedCall], List[PlannedCall]]:
    def _parse(l, market: bool = False) -> List[PlannedCall]:
        return [
            PlannedCall(
                x["name"], 
                x["method"], 
                x["path"], 
                x.get("path_params", {}) or {}, 
                x.get("query_params", {}) or {}, 
                market
            ) for x in (l or [])
        ]
        
    core = _parse(plan_yaml.get("plans", {}).get("default", []))
    market = []
    if cfg["ingestion"].get("enable_market_context"):
        market = _parse(plan_yaml.get("plans", {}).get("market_context", []), True)
    return core, market

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

def generate_predictions(
    cfg: Dict[str, Any],
    snapshot_id: int,
    valid_features: List[Dict[str, Any]],
    asof_utc: dt.datetime,
    session_enum: SessionState,
    sec_to_close: Optional[float],
    endpoint_coverage: float
) -> List[Dict[str, Any]]:
    """
    Centralized Decision Window and Gating pipeline.
    Shared identically between live ingestion and replay engine to guarantee governance parity.
    """
    feature_value_map = {f["feature_key"]: f["feature_value"] for f in valid_features}
    freshness_by_feature = {f["feature_key"]: f["meta_json"].get("freshness_state", "ERROR") for f in valid_features}
    stale_age_by_feature = {f["feature_key"]: f["meta_json"].get("stale_age_min") for f in valid_features}
    
    alignment_tolerance_sec = cfg["validation"]["alignment_tolerance_sec"]
    cadence_sec = int(cfg["ingestion"]["cadence_minutes"]) * 60
    
    ts_list = []
    alignment_violations = []
    missing_ts_features = []
    future_ts_features = []
    normalized_future_ts_features = []
    aligned_features = []
    
    for f in valid_features:
        meta = f["meta_json"]
        metric_lineage = meta.get("metric_lineage", {})
        eff_ts_str = metric_lineage.get("effective_ts_utc")
        
        if not isinstance(eff_ts_str, str) or not eff_ts_str:
            logger.warning(
                f"feature_missing_effective_ts: {f['feature_key']}", 
                extra={"counter": "feature_missing_effective_ts", "feature_key": f['feature_key']}
            )
            missing_ts_features.append(f['feature_key'])
            continue

        try:
            eff_ts = dt.datetime.fromisoformat(eff_ts_str.replace('Z', '+00:00'))
            if eff_ts.tzinfo is None:
                logger.warning(
                    f"feature_invalid_effective_ts (naive timezone): {f['feature_key']}", 
                    extra={"counter": "feature_invalid_effective_ts", "feature_key": f['feature_key']}
                )
                missing_ts_features.append(f['feature_key'])
                continue
        except Exception:
            logger.warning(
                f"feature_invalid_effective_ts (malformed): {f['feature_key']}", 
                extra={"counter": "feature_invalid_effective_ts", "feature_key": f['feature_key']}
            )
            missing_ts_features.append(f['feature_key'])
            continue

        delta_sec = (asof_utc - eff_ts).total_seconds()
        
        if delta_sec < 0:
            drift_sec = abs(delta_sec)
            if drift_sec < cadence_sec:
                logger.info(
                    f"normalized_future_ts: {f['feature_key']} timestamp {eff_ts.isoformat()} clamped to {asof_utc.isoformat()}",
                    extra={"counter": "normalized_future_ts_count", "feature_key": f['feature_key'], "drift_sec": int(drift_sec)}
                )
                eff_ts = asof_utc
                delta_sec = 0.0
                normalized_future_ts_features.append(f['feature_key'])
                
                # Task 7: Clamp-Lineage Consistency (Explainability Must Match Gating)
                if "metric_lineage" in meta:
                    meta["metric_lineage"]["effective_ts_utc"] = asof_utc.isoformat()
                if "details" not in meta:
                    meta["details"] = {}
                meta["details"]["clamped_future_ts"] = True
                
            else:
                logger.warning(
                    f"future_ts_violation: {f['feature_key']} is ahead of asof_utc by {int(drift_sec)}s", 
                    extra={
                        "counter": "future_ts_violation_count", 
                        "feature_key": f['feature_key'], 
                        "delta_sec": int(delta_sec)
                    }
                )
                future_ts_features.append(f['feature_key'])
                continue

        if delta_sec > alignment_tolerance_sec:
            logger.warning(
                f"alignment_violation: {f['feature_key']} misaligned by {int(delta_sec)}s", 
                extra={
                    "counter": "alignment_violation_count", 
                    "feature_key": f['feature_key'], 
                    "delta_sec": int(delta_sec)
                }
            )
            alignment_violations.append(f"{f['feature_key']}_delta_{int(delta_sec)}s")
            continue
            
        ts_list.append(eff_ts)
        aligned_features.append(f)

    feat_dict = {f["feature_key"]: f["feature_value"] for f in aligned_features}
    
    source_ts_min = min(ts_list) if ts_list else None
    source_ts_max = max(ts_list) if ts_list else None
    is_aligned = len(alignment_violations) == 0 and len(missing_ts_features) == 0 and len(future_ts_features) == 0

    base_gate = DecisionGate(
        data_quality_state=DataQualityState.VALID, 
        risk_gate_status=RiskGateStatus.PASS, 
        decision_state=SignalState.NEUTRAL
    )

    if session_enum == SessionState.CLOSED:
        base_gate = base_gate.block("session_closed", invalid=True)
        
    if not feat_dict and valid_features:
        base_gate = base_gate.block("all_features_excluded_by_alignment_or_ts", invalid=True)

    def evaluate_horizon_gate(h_str: str, current_base_gate: DecisionGate) -> Tuple[DecisionGate, Dict[str, float], float, Dict[str, Any], List[str], Dict[str, Any]]:
        old_dq_state = current_base_gate.data_quality_state
        
        base_reqs = cfg["validation"]["horizon_critical_features"][h_str]
        
        if cfg["validation"]["horizon_weights_source"] == "model":
            weights = dict(cfg["model"]["weights"])
            overrides = cfg["validation"]["horizon_weights_overrides"][h_str]
            if overrides:
                weights.update(overrides)
        else:
            weights = cfg["validation"]["horizon_weights"][h_str]
        
        use_default_reqs = cfg["validation"]["use_default_required_features"]
        
        reqs = list(base_reqs)
        if use_default_reqs:
            session_default_criticals = {
                SessionState.RTH: ["spot", "net_gex_sign", "smart_whale_pressure", "oi_pressure"],
                SessionState.PREMARKET: ["spot", "dealer_vanna"],
                SessionState.AFTERHOURS: ["spot"],
                SessionState.CLOSED: ["spot"]
            }.get(session_enum, ["spot"])
            reqs = list(set(reqs) | set(session_default_criticals))
        else:
            if not reqs:
                logger.info(f"No explicit critical features for horizon {h_str} and defaults disabled. Empty critical set allowed.")
                
        target_features = set(weights.keys()) | set(reqs)
        
        horizon_contract = {
            "use_default_required_features": use_default_reqs,
            "resolved_critical_features": sorted(list(reqs))
        }
        
        if not target_features:
            logger.error(f"invalid_contract_horizon_{h_str}: No target features (critical or weighted) defined.", extra={"counter": "invalid_horizon_contract"})
            return current_base_gate.block(f"invalid_contract_no_targets_{h_str}", invalid=True), weights, 0.0, horizon_contract, ["invalid_contract_no_targets"], {}
            
        logger.info(
            f"resolved_horizon_{h_str}_features",
            extra={
                "counter": "resolved_horizon_features",
                "horizon": h_str,
                "critical_features": sorted(list(reqs)),
                "target_features": sorted(list(target_features))
            }
        )
        
        valid_target_count = 0.0
        dq_reasons = []
        
        for k in target_features:
            if any(av.startswith(f"{k}_delta_") for av in alignment_violations):
                dq_reasons.append(f"{k}_misaligned")
            elif k in future_ts_features:
                dq_reasons.append(f"{k}_future_ts")
            elif k in missing_ts_features:
                dq_reasons.append(f"{k}_missing_ts")
            elif k not in feature_value_map or feature_value_map[k] is None or not math.isfinite(feature_value_map[k]):
                dq_reasons.append(f"{k}_missing_or_invalid")
            else:
                f_state = freshness_by_feature.get(k, "ERROR")
                if f_state in ("FRESH", "EMPTY_VALID"):
                    valid_target_count += 1.0
                elif f_state == "STALE_CARRY":
                    age = stale_age_by_feature.get(k)
                    if age is not None:
                        invalid_mins = cfg["validation"]["invalid_after_minutes"]
                        penalty_ratio = min(1.0, age / float(invalid_mins)) if invalid_mins > 0 else 1.0
                        credit = max(0.1, 1.0 - penalty_ratio)
                        valid_target_count += credit
                        dq_reasons.append(f"{k}_stale_carry_age_{int(age)}m")
                    else:
                        valid_target_count += 0.5
                        dq_reasons.append(f"{k}_stale_carry_fixed_penalty")
                        logger.warning(f"stale_age_missing_for_feature: {k}", extra={"counter": "stale_age_missing", "feature_key": k})
                else:
                    dq_reasons.append(f"{k}_bad_freshness_{f_state}")
                    
        decision_dq = valid_target_count / len(target_features) if target_features else 0.0
        
        if dq_reasons:
            logger.info(
                f"dq_reasons_horizon_{h_str}", 
                extra={"counter": "decision_dq_reasons", "horizon": h_str, "reasons": dq_reasons, "decision_dq": decision_dq}
            )
        
        missing_criticals = [k for k in reqs if k not in feat_dict or feat_dict[k] is None or not math.isfinite(feat_dict[k])]
        missing_non_criticals = [k for k in weights.keys() if k not in reqs and (k not in feat_dict or feat_dict[k] is None or not math.isfinite(feat_dict[k]))]
        
        valid_target_keys = {k for k in target_features if k in feat_dict and feat_dict[k] is not None and math.isfinite(feat_dict[k])}

        h_gate = current_base_gate
        target_diagnostics = {}
        
        # Task 8 & 10: Stronger Gate Diagnostics for no_horizon_target_features_after_alignment
        if len(valid_target_keys) == 0:
            expected_keys = sorted(list(target_features))
            filter_reasons = {
                "missing": sorted([k for k in expected_keys if k not in feature_value_map and not any(av.startswith(f"{k}_delta_") for av in alignment_violations) and k not in future_ts_features and k not in missing_ts_features]),
                "misaligned": sorted([k for k in expected_keys if any(av.startswith(f"{k}_delta_") for av in alignment_violations)]),
                "future_ts": sorted([k for k in expected_keys if k in future_ts_features]),
                "missing_ts": sorted([k for k in expected_keys if k in missing_ts_features]),
                "invalid": sorted([k for k in expected_keys if k in feature_value_map and (feature_value_map[k] is None or not math.isfinite(feature_value_map[k]))])
            }
            
            # Populate structured dictionary for meta_json storage
            target_diagnostics = {
                "expected_keys": expected_keys,
                "missing_keys": filter_reasons["missing"],
                "invalid_keys": filter_reasons["invalid"],
                "misaligned_keys": filter_reasons["misaligned"],
                "future_ts_keys": filter_reasons["future_ts"],
                "missing_ts_keys": filter_reasons["missing_ts"]
            }
            
            actual_missing_criticals = sorted([k for k in reqs if k not in feat_dict or feat_dict[k] is None or not math.isfinite(feat_dict[k])])
            excluded_align_count = len(filter_reasons["misaligned"]) + len(filter_reasons["future_ts"]) + len(filter_reasons["missing_ts"])
            
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
                    "filter_reasons": filter_reasons
                }
            )
            h_gate = h_gate.block(block_reason, invalid=True, missing_features=actual_missing_criticals)

        if missing_criticals and h_gate.risk_gate_status != RiskGateStatus.BLOCKED:
            missing_criticals = sorted(missing_criticals)
            logger.warning(
                f"no_signal_due_to_critical_missing_horizon_{h_str}", 
                extra={
                    "counter": "no_signal_due_to_critical_missing_count", 
                    "missing_features": missing_criticals,
                    "horizon": h_str
                }
            )
            reasons = []
            for mk in missing_criticals:
                if any(av.startswith(f"{mk}_delta_") for av in alignment_violations):
                    reason_str = f"{mk}_misaligned"
                elif mk in future_ts_features:
                    reason_str = f"{mk}_future_ts"
                elif mk in missing_ts_features:
                    reason_str = f"{mk}_missing_ts"
                else:
                    reason_str = f"{mk}_missing"
                reasons.append(reason_str)
                
                logger.warning(
                    f"critical_feature_failed_{h_str}: {reason_str}", 
                    extra={"counter": "critical_feature_missing_count", "feature_key": mk, "horizon": h_str, "reason": reason_str}
                )
            h_gate = h_gate.block(f"critical_features_failed_{h_str}: {','.join(reasons)}", invalid=True, missing_features=missing_criticals)
        elif missing_non_criticals and h_gate.risk_gate_status == RiskGateStatus.PASS:
            reasons = []
            for mk in missing_non_criticals:
                if any(av.startswith(f"{mk}_delta_") for av in alignment_violations):
                    reason_str = f"{mk}_misaligned"
                elif mk in future_ts_features:
                    reason_str = f"{mk}_future_ts"
                elif mk in missing_ts_features:
                    reason_str = f"{mk}_missing_ts"
                else:
                    reason_str = f"{mk}_missing"
                reasons.append(reason_str)
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
                    "risk_status": h_gate.risk_gate_status.value
                }
            )

        return h_gate, weights, decision_dq, horizon_contract, sorted(dq_reasons), target_diagnostics

    predictions = []

    for h in cfg["validation"]["horizons_minutes"]:
        h_str = str(h)
        h_gate, weights, decision_dq, horizon_contract, dq_reasons, target_diags = evaluate_horizon_gate(h_str, base_gate)
        pred = bounded_additive_score(feat_dict, decision_dq, weights, gate=h_gate)
        
        window_id_fixed = hashlib.sha256(f"{snapshot_id}_FIXED_{h}".encode()).hexdigest()[:16]
        
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
            "normalized_keys": normalized_future_ts_features
        }
        pred_meta["horizon_contract"] = horizon_contract
        pred_meta["decision_dq"] = decision_dq
        pred_meta["dq_reason_codes"] = dq_reasons
        
        # Task 10: Store structured gate diagnostics
        pred_meta["gating"] = {"target_diagnostics": target_diags} if target_diags else {}
        
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
            "decision_window_id": window_id_fixed
        })

    emit_to_close = cfg["validation"]["emit_to_close_horizon"]
    
    if emit_to_close and sec_to_close is not None and sec_to_close > 0:
        h_gate, weights, decision_dq, horizon_contract, dq_reasons, target_diags = evaluate_horizon_gate("to_close", base_gate)
        pred = bounded_additive_score(feat_dict, decision_dq, weights, gate=h_gate)
        
        window_id_close = hashlib.sha256(f"{snapshot_id}_TOCLOSE_{sec_to_close}".encode()).hexdigest()[:16]
        
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
            "normalized_keys": normalized_future_ts_features
        }
        pred_meta["horizon_contract"] = horizon_contract
        pred_meta["decision_dq"] = decision_dq
        pred_meta["dq_reason_codes"] = dq_reasons
        
        # Task 10: Store structured gate diagnostics
        pred_meta["gating"] = {"target_diagnostics": target_diags} if target_diags else {}
        
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
            "decision_window_id": window_id_close
        })

    return predictions

def _ingest_once_impl(cfg: Dict[str, Any], catalog_path: str, config_path: str) -> None:
    _validate_config(cfg)
    registry = load_api_catalog(catalog_path)
    plan_yaml = load_endpoint_plan("src/config/endpoint_plan.yaml")
    
    validate_plan_coverage(plan_yaml)
    
    core, market = build_plan(cfg, plan_yaml)
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
    
    canonical_labels = {
        SessionState.PREMARKET.value,
        SessionState.RTH.value,
        SessionState.AFTERHOURS.value,
        SessionState.CLOSED.value
    }
    
    if sess_str not in canonical_labels:
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
        
    session_enum = SessionState(sess_str)
        
    close_utc = hours.market_close_et.astimezone(UTC) if hours.market_close_et else None
    post_utc = hours.post_end_et.astimezone(UTC) if hours.post_end_et else None
    sec_to_close = hours.seconds_to_close(asof_et)

    async def _run_fetch():
        net = cfg.get("network", {})
        sys_cfg = cfg.get("system", {})
        cb = net.get("circuit_breaker", {})
        
        async with UwClient(
            registry=registry, 
            base_url=net.get("base_url", "https://api.unusualwhales.com"),
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
                    
                    event_id = db.insert_raw_event(
                        con, run_id, tkr, endpoint_id, res.requested_at_utc, res.received_at_utc,
                        res.status_code, res.latency_ms, res.payload_hash, res.payload_json,
                        res.retry_count > 0, res.error_type, res.error_message, cb,
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
                        invalid_after_seconds=invalid_after_seconds
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
                            effective_ts_utc=res.effective_ts_utc
                        )
                        contexts[endpoint_id] = ctx
                            
                        src_meta = {
                            "method": call.method,
                            "path": call.path,
                            "operation_id": op_id,
                            "endpoint_id": endpoint_id,
                            "signature": sig,
                            "used_event_id": res.used_event_id,
                            "missing_keys": asmnt.missing_keys
                        }

                        # Ticket 3: Persist strict validation keys for Replay Parity constraint
                        lineage_meta = MetaContract(
                            source_endpoints=[src_meta],
                            freshness_state=f_state.name,
                            stale_age_min=ctx.stale_age_min,
                            na_reason=n_reason,
                            details={
                                "effective_ts_utc": ctx.effective_ts_utc.isoformat() if ctx.effective_ts_utc else None,
                                "endpoint_asof_ts_utc": ctx.endpoint_asof_ts_utc.isoformat() if ctx.endpoint_asof_ts_utc else None,
                                "alignment_delta_sec": ctx.alignment_delta_sec,
                                "truth_status": res.payload_class.name if hasattr(res.payload_class, "name") else str(res.payload_class),
                                "stale_age_seconds": res.stale_age_seconds
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
                                logger.warning(
                                    f"Malformed feature row (non-finite value): {f}", 
                                    extra={"counter": "malformed_rows_dropped"}
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
                        
                        logger.warning(
                            f"Malformed feature row skipped: {f}", 
                            extra={"counter": "malformed_rows_dropped"}
                        )
                        malformed_count += 1

                    for l in levels_insert_list:
                        if isinstance(l, dict) and "level_type" in l and "meta_json" in l:
                            p = l.get("price")
                            m = l.get("magnitude")
                            
                            if (p is not None and not _is_valid_num(p)) or (m is not None and not _is_valid_num(m)):
                                logger.warning(
                                    f"Malformed level row (non-finite value): {l}", 
                                    extra={"counter": "malformed_rows_dropped"}
                                )
                                malformed_count += 1
                                continue
                                
                            meta = l["meta_json"]
                            if isinstance(meta, dict) and all(k in meta for k in ["source_endpoints", "freshness_state", "stale_age_min", "na_reason", "details", "metric_lineage"]):
                                valid_levels.append(l) 
                                continue
                        
                        logger.warning(
                            f"Malformed level row skipped: {l}", 
                            extra={"counter": "malformed_rows_dropped"}
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
                        endpoint_coverage=endpoint_coverage
                    )
                    
                    # Task 7: Persist the features *after* prediction gating mutations.
                    # This guarantees the stored JSON strictly matches what the gate utilized.
                    db.insert_features(con, snapshot_id, valid_features)
                    db.insert_levels(con, snapshot_id, valid_levels)
                    
                    for p in predictions:
                        db.insert_prediction(con, p)

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