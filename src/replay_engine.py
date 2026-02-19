import json
import duckdb
import pandas as pd
from src.config_loader import load_yaml
from src import features as feat

def run_replay(ticker: str):
    cfg = load_yaml("src/config/config.yaml").raw
    con = duckdb.connect(cfg["storage"]["duckdb_path"], read_only=True)
    
    print(f"--- REPLAYING {ticker} ---")
    endpoints = con.execute("SELECT endpoint_id, path FROM dim_endpoints").fetchdf()
    id_map = dict(zip(endpoints['endpoint_id'], endpoints['path']))

    rows = con.execute("SELECT received_at_utc, endpoint_id, payload_json FROM raw_http_events WHERE ticker = ? ORDER BY received_at_utc ASC", [ticker.upper()]).fetchall()
    
    cycles = {}
    for ts, eid, p_str in rows:
        if ts not in cycles: cycles[ts] = {}
        path = id_map.get(eid, "")
        # Route payloads based on URL substring
        if "ohlc" in path: cycles[ts]["ohlc"] = json.loads(p_str)
        if "flow-recent" in path: cycles[ts]["flow"] = json.loads(p_str)
        if "greek-exposure" in path: cycles[ts]["greeks"] = json.loads(p_str)

    results = []
    for ts, payloads in cycles.items():
        spot = feat.extract_price_features(payloads.get("ohlc")).features.get("spot")
        whale = feat.extract_smart_whale_pressure(payloads.get("flow"))
        vanna = feat.extract_dealer_greeks(payloads.get("greeks"))
        
        results.append({
            "Time": ts, 
            "Spot": spot,
            "Whale": whale.features.get("smart_whale_pressure", 0.0), 
            "Vanna": vanna.features.get("dealer_vanna", 0.0)
        })

    df = pd.DataFrame(results)
    activity = df[(df["Whale"] != 0) | (df["Vanna"] != 0)]
    if activity.empty:
        print("No signals found. Check if your database contains 'flow-recent' and 'greek-exposure' payloads.")
    else:
        print(activity)

if __name__ == "__main__":
    import sys
    run_replay(sys.argv[1] if len(sys.argv) > 1 else "TSLA")