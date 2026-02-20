import duckdb
import pandas as pd
import json
from typing import Optional
from src.config_loader import load_yaml
from src import features as feat
from src.endpoint_truth import EndpointContext

def run_replay(db_path: str, ticker: str, start_ts: Optional[str] = None, end_ts: Optional[str] = None):
    con = duckdb.connect(db_path, read_only=True)
    print(f"--- REPLAYING {ticker.upper()} ---")
    
    query = "SELECT snapshot_id, asof_ts_utc FROM snapshots WHERE ticker = ?"
    params = [ticker.upper()]
    if start_ts:
        query += " AND asof_ts_utc >= ?"; params.append(start_ts)
    if end_ts:
        query += " AND asof_ts_utc <= ?"; params.append(end_ts)
        
    query += " ORDER BY asof_ts_utc ASC"
    snapshots = con.execute(query, params).fetchall()
    
    results = []
    for snap_id, asof_ts in snapshots:
        lineage_rows = con.execute(
            "SELECT endpoint_id, used_event_id, freshness_state, data_age_seconds, na_reason, payload_class "
            "FROM snapshot_lineage WHERE snapshot_id = ?", [str(snap_id)]
        ).fetchall()
        
        effective_payloads, contexts = {}, {}
        for eid, used_eid, f_state, age, na_reason, p_class in lineage_rows:
            if used_eid:
                pj_row = con.execute("SELECT payload_json FROM raw_http_events WHERE event_id = ?", [str(used_eid)]).fetchone()
                if pj_row and pj_row[0] is not None:
                    try:
                        effective_payloads[eid] = json.loads(pj_row[0]) if isinstance(pj_row[0], str) else pj_row[0]
                    except Exception:
                        effective_payloads[eid] = None
                        na_reason = "missing_raw_payload_for_lineage"
                else:
                    effective_payloads[eid] = None
                    na_reason = "missing_raw_payload_for_lineage"
            else:
                effective_payloads[eid] = None
                
            ep_info = con.execute("SELECT method, path, signature FROM dim_endpoints WHERE endpoint_id = ?", [eid]).fetchone()
            if ep_info:
                contexts[eid] = EndpointContext(
                    endpoint_id=eid, method=ep_info[0], path=ep_info[1], operation_id=None, signature=ep_info[2],
                    used_event_id=used_eid, payload_class=p_class, freshness_state=f_state,
                    stale_age_min=(age // 60) if age is not None else None, na_reason=na_reason
                )
                
        # Use exact feature extraction routing matching ingest
        f_rows, _ = feat.extract_all(effective_payloads, contexts)
        feat_dict = {f["feature_key"]: f["feature_value"] for f in f_rows}
        
        # Build pure dictionary representation allowing None natively
        results.append({
            "Time": asof_ts,
            "Spot": feat_dict.get("spot"),
            "Whale": feat_dict.get("smart_whale_pressure"),
            "Vanna": feat_dict.get("dealer_vanna")
        })

    df = pd.DataFrame(results)
    if df.empty:
        print("No snapshots found.")
    else:
        # Display purely matching data
        print(df.to_string())