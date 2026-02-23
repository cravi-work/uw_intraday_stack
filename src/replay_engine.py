import duckdb
import json
import math
from typing import Optional
from dataclasses import asdict
from src.config_loader import load_yaml
from src import features as feat
from src.endpoint_truth import EndpointContext
from src.models import bounded_additive_score, DecisionGate
from datetime import datetime, timezone

def run_replay(db_path: str, ticker: str, start_ts: Optional[str] = None, end_ts: Optional[str] = None):
    cfg = load_yaml("src/config/config.yaml").raw
    con = duckdb.connect(db_path, read_only=True)
    print(f"--- REPLAY PARITY CHECK: {ticker.upper()} ---")
    
    query = "SELECT snapshot_id, asof_ts_utc, data_quality_score FROM snapshots WHERE ticker = ?"
    params = [ticker.upper()]
    if start_ts: query += " AND asof_ts_utc >= ?"; params.append(start_ts)
    if end_ts: query += " AND asof_ts_utc <= ?"; params.append(end_ts)
        
    query += " ORDER BY asof_ts_utc ASC"
    snapshots = con.execute(query, params).fetchall()
    
    for snap_id, asof_ts, dq in snapshots:
        lineage_rows = con.execute("SELECT endpoint_id, used_event_id, freshness_state, data_age_seconds, na_reason, payload_class FROM snapshot_lineage WHERE snapshot_id = ?", [str(snap_id)]).fetchall()
        
        used_event_ids = [str(r[1]) for r in lineage_rows if r[1] is not None]
        payloads_dict = {}
        if used_event_ids:
            placeholders = ','.join(['?'] * len(used_event_ids))
            raw_rows = con.execute(f"SELECT event_id, payload_json FROM raw_http_events WHERE event_id IN ({placeholders})", used_event_ids).fetchall()
            for r_id, p_json in raw_rows:
                if p_json is not None:
                    try: payloads_dict[str(r_id)] = json.loads(p_json) if isinstance(p_json, str) else p_json
                    except json.JSONDecodeError as e: raise RuntimeError(f"JSON Parse Error for event {r_id}: {e}")

        effective_payloads, contexts = {}, {}
        for eid, used_eid, f_state, age, na_reason, p_class in lineage_rows:
            if used_eid:
                p_val = payloads_dict.get(str(used_eid))
                effective_payloads[eid] = p_val
                if p_val is None: na_reason = "missing_raw_payload_for_lineage"
            else:
                effective_payloads[eid] = None
                
            ep_info = con.execute("SELECT method, path, signature FROM dim_endpoints WHERE endpoint_id = ?", [eid]).fetchone()
            if ep_info:
                if asof_ts.tzinfo is None: asof_ts = asof_ts.replace(tzinfo=timezone.utc)
                delta_sec = age if age is not None else 0
                ep_asof = asof_ts - __import__('datetime').timedelta(seconds=delta_sec)

                contexts[eid] = EndpointContext(
                    endpoint_id=eid, method=ep_info[0], path=ep_info[1], operation_id=None, signature=ep_info[2],
                    used_event_id=str(used_eid) if used_eid else None, payload_class=p_class, freshness_state=f_state,
                    stale_age_min=(age // 60) if age is not None else None, na_reason=na_reason,
                    endpoint_asof_ts_utc=ep_asof, alignment_delta_sec=delta_sec
                )
                
        f_rows, l_rows = feat.extract_all(effective_payloads, contexts)
        recomputed_features = {f["feature_key"]: f["feature_value"] for f in f_rows}
        
        stored_f_rows = con.execute("SELECT feature_key, feature_value FROM features WHERE snapshot_id = ?", [str(snap_id)]).fetchall()
        stored_features = {r[0]: r[1] for r in stored_f_rows}

        for k, recomputed_v in recomputed_features.items():
            stored_v = stored_features.get(k)
            if stored_v is None and recomputed_v is None: continue
            if (stored_v is None) != (recomputed_v is None) or not math.isclose(stored_v, recomputed_v, abs_tol=1e-9):
                raise RuntimeError(f"PARITY MISMATCH: Snapshot {snap_id} Feature '{k}' -> Stored: {stored_v} | Recomputed: {recomputed_v}")

        stored_preds = con.execute("SELECT horizon_kind, horizon_minutes, horizon_seconds, decision_state, risk_gate_status, prob_up, prob_down, prob_flat FROM predictions WHERE snapshot_id = ?", [str(snap_id)]).fetchall()
        
        if stored_preds:
            base_gate = DecisionGate(data_quality_state="VALID", risk_gate_status="PASS", decision_state="NEUTRAL")
            start_price = recomputed_features.get("spot")

            for ctx in contexts.values():
                if ctx.alignment_delta_sec is not None and ctx.alignment_delta_sec > int(cfg.get("validation", {}).get("alignment_tolerance_sec", 1800)):
                    base_gate = base_gate.degrade(f"alignment_tolerance_exceeded_ep_{ctx.endpoint_id}", partial=True)

            if start_price is None:
                base_gate = base_gate.block("missing_critical_feature_spot", invalid=True)
            elif dq < 0.5:
                base_gate = base_gate.degrade(f"low_data_quality_score_{dq:.2f}", partial=True)

            weights = cfg.get("validation", {}).get("model_weights", {"smart_whale_pressure": 1.0, "net_gex_sign": 0.5, "dealer_vanna": 0.5})

            for sp_row in stored_preds:
                pred = bounded_additive_score(recomputed_features, dq, weights, gate=base_gate)
                if sp_row[3] != pred.gate.decision_state or sp_row[4] != pred.gate.risk_gate_status:
                    raise RuntimeError(f"PARITY MISMATCH: Snapshot {snap_id} Risk/Decision Governance altered. Stored: {sp_row[3]}/{sp_row[4]} | Recomputed: {pred.gate.decision_state}/{pred.gate.risk_gate_status}")

    print("✅ Replay Parity Check Passed: Recomputed logic exactly matches stored features & decision governance across all horizons.")