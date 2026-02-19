import duckdb

DB = "uw_stack.duckdb"

SQL = [
  "CREATE SEQUENCE IF NOT EXISTS seq_endpoint_id",
  "CREATE SEQUENCE IF NOT EXISTS seq_config_version",

  "CREATE TABLE IF NOT EXISTS dim_endpoints(endpoint_id INTEGER PRIMARY KEY DEFAULT nextval('seq_endpoint_id'), method TEXT, path TEXT, params_hash TEXT, signature TEXT UNIQUE)",

  "CREATE TABLE IF NOT EXISTS raw_http_events(event_id UUID PRIMARY KEY, received_at_utc TIMESTAMP, ticker TEXT, endpoint_id INTEGER, http_status INTEGER, payload_json JSON, payload_hash TEXT, error_msg TEXT, latency_ms INTEGER)",

  "CREATE TABLE IF NOT EXISTS snapshots(snapshot_id UUID PRIMARY KEY, asof_ts_utc TIMESTAMP, ticker TEXT, session_label TEXT, UNIQUE(ticker, asof_ts_utc))",

  "CREATE TABLE IF NOT EXISTS snapshot_lineage(snapshot_id UUID, endpoint_id INTEGER, used_event_id UUID, freshness_state TEXT, data_age_seconds INTEGER, PRIMARY KEY(snapshot_id, endpoint_id))",

  "CREATE TABLE IF NOT EXISTS features(snapshot_id UUID PRIMARY KEY, features_json JSON, created_at_utc TIMESTAMP)",

  "CREATE TABLE IF NOT EXISTS levels(snapshot_id UUID, level_type TEXT, price DOUBLE, magnitude DOUBLE, meta_json JSON)",

  "CREATE TABLE IF NOT EXISTS predictions(prediction_id UUID PRIMARY KEY, snapshot_id UUID, horizon_minutes INTEGER, bias TEXT, confidence DOUBLE, prob_up DOUBLE, prob_down DOUBLE, start_price DOUBLE, created_at_utc TIMESTAMP, outcome_realized BOOLEAN DEFAULT FALSE, outcome_label TEXT, outcome_price DOUBLE, is_correct BOOLEAN, brier_score DOUBLE, log_loss DOUBLE)",

  "CREATE TABLE IF NOT EXISTS meta_config(config_version INTEGER PRIMARY KEY DEFAULT nextval('seq_config_version'), config_hash TEXT UNIQUE, config_yaml TEXT, created_at_utc TIMESTAMP)",

  "CREATE TABLE IF NOT EXISTS adapter_actions(action_id UUID PRIMARY KEY, run_id UUID, from_config_version INTEGER, to_config_version INTEGER, reason TEXT, diff_json TEXT, created_at_utc TIMESTAMP)"
]

con = duckdb.connect(DB)
for stmt in SQL:
  con.execute(stmt)
con.close()

print("Schema initialized in", DB)
