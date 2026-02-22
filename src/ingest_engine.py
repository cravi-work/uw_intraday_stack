from __future__ import annotations

import asyncio
import datetime as dt
import logging
import math
import uuid
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
from .models import bounded_additive_score, Prediction, DecisionGate
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
    if "horizons_minutes" not in cfg["validation"]:
        raise KeyError("Missing validation.horizons_minutes")

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
    """Helper to ensure derived outputs are strictly mathematically finite."""
    return isinstance(v, (int, float)) and math.isfinite(v)

def _ingest_once_impl(cfg: Dict[str, Any], catalog_path: str, config_path: str) -> None:
    _validate_config(cfg)
    registry = load_api_catalog(catalog_path)
    plan_yaml = load_endpoint_plan("src/config/endpoint_plan.yaml")
    
    validate_plan_coverage(plan_yaml)
    
    core, market = build_plan(cfg, plan_yaml)
    tickers = [t.upper() for t in cfg["ingestion"]["watchlist"]]

    val_cfg = cfg.get("validation", {})
    fallback_max_age_seconds = int(val_cfg.get("fallback_max_age_minutes", 15)) * 60
    invalid_after_seconds = int(val_cfg.get("invalid_after_minutes", 60)) * 60

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

    sess = hours.get_session_label(asof_et)
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
    
    # Pre-sort fetches to deterministically handle out-of-order execution packets within identical run cycles
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

                run_notes = f"SESS={sess}"
                if hours.reason != "NORMAL":
                    run_notes += f"; {hours.reason}"
                    
                run_id = db.begin_run(
                    con, asof_utc, sess, hours.is_trading_day, 
                    hours.is_early_close, cfg_ver, registry.catalog_hash, notes=run_notes
                )

                events_by_ticker: Dict[str, List[Tuple[int, uuid.UUID, Any, PlannedCall, str, PayloadAssessment]]] = {t: [] for t in tickers}
                max_seen_ts = {}

                for (tkr, call, sig, qp, res, cb) in fetch_results:
                    endpoint_id = db.upsert_endpoint(con, call.method, call.path, qp, registry)
                    
                    # Deterministic Ordering: Drop Out-Of-Order packets from mutating system state
                    ev_key = (tkr, endpoint_id)
                    if ev_key in max_seen_ts and res.requested_at_utc < max_seen_ts[ev_key]:
                        logger.warning(f"Out of order packet dropped from state mutation: {ev_key}")
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
                    
                    assessment = classify_payload(res, prev_hash, call.method, call.path, sess)

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
                        str(event_id), attempt_ts_utc, assessment, prev_state,
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

                    if assessment.error_reason and not enforced_reason:
                        enforced_reason = assessment.error_reason

                    resolved = replace(resolved, freshness_state=enforced_freshness, na_reason=enforced_reason)

                    if tkr not in events_by_ticker:
                        events_by_ticker[tkr] = []
                        
                    events_by_ticker[tkr].append((endpoint_id, event_id, resolved, call, sig, assessment))

                for tkr in tickers:
                    evs = events_by_ticker.get(tkr, [])
                    
                    valid_count = sum(1 for _, _, res, _, _, asmnt in evs if res.freshness_state == FreshnessState.FRESH or (res.freshness_state == FreshnessState.STALE_CARRY and res.used_event_id is not None) or (res.freshness_state == FreshnessState.EMPTY_VALID and asmnt.empty_policy.name == "EMPTY_IS_DATA"))
                    dq = (valid_count / len(evs)) if evs else 0.0

                    snapshot_id = db.insert_snapshot(
                        con, run_id=run_id, asof_ts_utc=asof_utc, ticker=tkr, session_label=sess,
                        is_trading_day=True, is_early_close=hours.is_early_close, data_quality_score=dq,
                        market_close_utc=close_utc, post_end_utc=post_utc, seconds_to_close=sec_to_close,
                    )
                    
                    active_used_ids = [
                        str(res.used_event_id) for _, _, res, _, _, _ in evs 
                        if res.used_event_id and res.freshness_state in (FreshnessState.FRESH, FreshnessState.STALE_CARRY, FreshnessState.EMPTY_VALID)
                    ]
                    
                    payloads_from_db = db.get_payloads_by_event_ids(con, active_used_ids)

                    effective_payloads: Dict[int, Any] = {}
                    contexts: Dict[int, EndpointContext] = {}

                    source_endpoints = []
                    freshness_states = []
                    ages = []
                    
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
                            na_reason=n_reason
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
                        
                        source_endpoints.append(src_meta)
                        freshness_states.append(f_state)
                        
                        if res.stale_age_seconds is not None:
                            ages.append(res.stale_age_seconds)

                        lineage_meta = MetaContract(
                            source_endpoints=[src_meta],
                            freshness_state=f_state.name,
                            stale_age_min=ctx.stale_age_min,
                            na_reason=n_reason,
                            details={}
                        )

                        db.insert_lineage(
                            con, snapshot_id=snapshot_id, endpoint_id=endpoint_id, used_event_id=res.used_event_id,
                            freshness_state=f_state.name, data_age_seconds=res.stale_age_seconds,
                            payload_class=res.payload_class.name, na_reason=n_reason, meta_json=asdict(lineage_meta)
                        )

                    features_insert_list, levels_insert_list = extract_all(effective_payloads, contexts)
                    
                    # --- Strict Validation Gate ---
                    valid_features = []
                    valid_levels = []
                    malformed_count = 0
                    seen_keys = set()
                    
                    for f in features_insert_list:
                        if isinstance(f, dict) and "feature_key" in f and "meta_json" in f:
                            f_key = f["feature_key"]
                            f_val = f.get("feature_value")
                            
                            if f_val is not None and not _is_valid_num(f_val):
                                logger.warning(f"Malformed feature row (non-finite value): {f}")
                                malformed_count += 1
                                continue
                                
                            if f_key in seen_keys:
                                raise RuntimeError(f"Duplicate feature key detected in insert list: {f_key}")
                            seen_keys.add(f_key)
                                
                            meta = f["meta_json"]
                            if isinstance(meta, dict) and all(k in meta for k in ["source_endpoints", "freshness_state", "stale_age_min", "na_reason", "details"]):
                                valid_features.append(f)
                                continue
                        
                        logger.warning(f"Malformed feature row skipped: {f}")
                        malformed_count += 1

                    for l in levels_insert_list:
                        if isinstance(l, dict) and "level_type" in l and "meta_json" in l:
                            p = l.get("price")
                            m = l.get("magnitude")
                            
                            if (p is not None and not _is_valid_num(p)) or (m is not None and not _is_valid_num(m)):
                                logger.warning(f"Malformed level row (non-finite value): {l}")
                                malformed_count += 1
                                continue
                                
                            meta = l["meta_json"]
                            if isinstance(meta, dict) and all(k in meta for k in ["source_endpoints", "freshness_state", "stale_age_min", "na_reason", "details"]):
                                valid_levels.append(l) 
                                continue
                        
                        logger.warning(f"Malformed level row skipped: {l}")
                        malformed_count += 1

                    total_outputs = len(features_insert_list) + len(levels_insert_list)
                    if total_outputs > 0 and (malformed_count / total_outputs) > 0.2:
                        logger.error(f"Extraction failed: {malformed_count}/{total_outputs} rows malformed (>20% threshold). Rollback enforced.")
                        raise RuntimeError(f"Extraction failed: {malformed_count}/{total_outputs} rows malformed.")
                    
                    db.insert_features(con, snapshot_id, valid_features)
                    db.insert_levels(con, snapshot_id, valid_levels)
                    
                    # --- Prediction Execution (Risk Gate Integrated) ---
                    feat_dict = {f["feature_key"]: f["feature_value"] for f in valid_features}
                    start_price = feat_dict.get("spot") 

                    gate = DecisionGate(data_quality_state="VALID", risk_gate_status="PASS", decision_state="NEUTRAL")

                    if start_price is None:
                        gate.data_quality_state = "INVALID"
                        gate.risk_gate_status = "BLOCKED"
                        gate.decision_state = "NO_TRADE"
                        gate.blocked_reasons.append("missing_critical_feature_spot")
                        gate.validation_eligible = False
                    elif dq < 0.5:
                        gate.data_quality_state = "PARTIAL"
                        gate.risk_gate_status = "DEGRADED"
                        gate.degraded_reasons.append(f"low_data_quality_score_{dq:.2f}")

                    weights = cfg.get("validation", {}).get("model_weights", {"smart_whale_pressure": 1.0, "net_gex_sign": 0.5, "dealer_vanna": 0.5})
                    pred = bounded_additive_score(feat_dict, dq, weights, gate=gate)

                    for h in cfg["validation"]["horizons_minutes"]:
                        db.insert_prediction(
                            con,
                            {
                                "snapshot_id": snapshot_id, "horizon_minutes": int(h), "horizon_kind": "FIXED", 
                                "horizon_seconds": None, "start_price": start_price, 
                                "bias": pred.bias, "confidence": pred.confidence, 
                                "prob_up": pred.prob_up, "prob_down": pred.prob_down, "prob_flat": pred.prob_flat,
                                "model_name": pred.model_name, "model_version": pred.model_version, "model_hash": pred.model_hash,
                                "is_mock": not pred.gate.validation_eligible, "meta_json": pred.meta,
                                "decision_state": pred.gate.decision_state, "risk_gate_status": pred.gate.risk_gate_status,
                                "data_quality_state": pred.gate.data_quality_state, "blocked_reasons": pred.gate.blocked_reasons,
                                "degraded_reasons": pred.gate.degraded_reasons, "validation_eligible": pred.gate.validation_eligible
                            }
                        )

                    if sec_to_close is not None and sec_to_close > 0:
                        db.insert_prediction(
                            con,
                            {
                                "snapshot_id": snapshot_id, "horizon_minutes": 0, "horizon_kind": "TO_CLOSE", 
                                "horizon_seconds": int(sec_to_close), "start_price": start_price, 
                                "bias": pred.bias, "confidence": pred.confidence, 
                                "prob_up": pred.prob_up, "prob_down": pred.prob_down, "prob_flat": pred.prob_flat,
                                "model_name": pred.model_name, "model_version": pred.model_version, "model_hash": pred.model_hash,
                                "is_mock": not pred.gate.validation_eligible, "meta_json": pred.meta,
                                "decision_state": pred.gate.decision_state, "risk_gate_status": pred.gate.risk_gate_status,
                                "data_quality_state": pred.gate.data_quality_state, "blocked_reasons": pred.gate.blocked_reasons,
                                "degraded_reasons": pred.gate.degraded_reasons, "validation_eligible": pred.gate.validation_eligible
                            }
                        )

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