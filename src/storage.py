from __future__ import annotations
import contextlib
import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional, List
import duckdb

from .api_catalog_loader import EndpointRegistry
from .endpoint_truth import to_utc_dt

logger = logging.getLogger(__name__)
UTC = timezone.utc

class DbLockError(RuntimeError): 
    pass

SCHEMA_SQL = """
CREATE SEQUENCE IF NOT EXISTS seq_endpoint_id START 1;
CREATE SEQUENCE IF NOT EXISTS seq_config_version START 1;

CREATE TABLE IF NOT EXISTS meta_runs (run_id UUID PRIMARY KEY, started_at_utc TIMESTAMP NOT NULL, ended_at_utc TIMESTAMP, asof_ts_utc TIMESTAMP NOT NULL, session_label TEXT, is_trading_day BOOLEAN, is_early_close BOOLEAN, config_version INTEGER, api_catalog_hash TEXT, notes TEXT);
CREATE TABLE IF NOT EXISTS snapshots (snapshot_id UUID PRIMARY KEY, run_id UUID REFERENCES meta_runs(run_id), asof_ts_utc TIMESTAMP NOT NULL, ticker TEXT NOT NULL, session_label TEXT, is_trading_day BOOLEAN, is_early_close BOOLEAN, data_quality_score DOUBLE, market_close_utc TIMESTAMP, post_end_utc TIMESTAMP, seconds_to_close INTEGER, created_at_utc TIMESTAMP DEFAULT current_timestamp, UNIQUE(ticker, asof_ts_utc));
CREATE TABLE IF NOT EXISTS meta_config (config_version INTEGER PRIMARY KEY, config_hash TEXT NOT NULL, config_yaml TEXT NOT NULL, created_at_utc TIMESTAMP DEFAULT current_timestamp);
CREATE TABLE IF NOT EXISTS dim_endpoints (endpoint_id INTEGER PRIMARY KEY, method TEXT, path TEXT, signature TEXT UNIQUE, params_hash TEXT, params_json JSON);
CREATE TABLE IF NOT EXISTS predictions (prediction_id UUID PRIMARY KEY, snapshot_id UUID REFERENCES snapshots(snapshot_id), horizon_minutes INTEGER, horizon_kind TEXT DEFAULT 'FIXED', horizon_seconds INTEGER, start_price DOUBLE, bias TEXT, confidence DOUBLE, prob_up DOUBLE, prob_down DOUBLE, prob_flat DOUBLE, model_name TEXT, model_version TEXT, model_hash TEXT, is_mock BOOLEAN DEFAULT FALSE, outcome_realized BOOLEAN DEFAULT FALSE, realized_at_utc TIMESTAMP, outcome_label TEXT, brier_score DOUBLE, log_loss DOUBLE, meta_json JSON);
CREATE UNIQUE INDEX IF NOT EXISTS idx_preds_dedupe ON predictions (snapshot_id, horizon_kind, horizon_minutes, horizon_seconds);

CREATE TABLE IF NOT EXISTS features (snapshot_id UUID REFERENCES snapshots(snapshot_id), feature_key TEXT, feature_value DOUBLE, meta_json JSON);
CREATE TABLE IF NOT EXISTS derived_levels (snapshot_id UUID, level_type TEXT, price DOUBLE, magnitude DOUBLE, meta_json JSON);
CREATE TABLE IF NOT EXISTS snapshot_lineage (snapshot_id UUID, endpoint_id INTEGER, used_event_id UUID, freshness_state TEXT, data_age_seconds INTEGER, payload_class TEXT, na_reason TEXT, meta_json JSON);
CREATE TABLE IF NOT EXISTS endpoint_state (ticker TEXT, endpoint_id INTEGER, last_success_event_id UUID, last_success_ts_utc TIMESTAMP, last_payload_hash TEXT, last_change_ts_utc TIMESTAMP, last_change_event_id UUID, last_attempt_event_id UUID, last_attempt_ts_utc TIMESTAMP, last_attempt_http_status INTEGER, last_attempt_error_type TEXT, last_attempt_error_msg TEXT, PRIMARY KEY (ticker, endpoint_id));
CREATE TABLE IF NOT EXISTS raw_http_events (event_id UUID PRIMARY KEY, run_id UUID, requested_at_utc TIMESTAMP, received_at_utc TIMESTAMP, ticker TEXT, endpoint_id INTEGER, http_status INTEGER, latency_ms INTEGER, payload_hash TEXT, payload_json JSON, is_retry BOOLEAN, error_type TEXT, error_msg TEXT, circuit_state_json JSON);
CREATE TABLE IF NOT EXISTS config_history (config_version VARCHAR, ingested_at_utc TIMESTAMP, yaml_content VARCHAR);
CREATE TABLE IF NOT EXISTS dim_tickers (ticker TEXT PRIMARY KEY);
"""

class DbWriter:
    def __init__(self, duckdb_path: str, lock_path: str = "uw.lock"):
        self.duckdb_path = duckdb_path
        self.lock_path = lock_path

    def _connect_new(self) -> duckdb.DuckDBPyConnection:
        con = duckdb.connect(self.duckdb_path)
        con.execute("PRAGMA threads=4")
        return con

    def ensure_schema(self, con: duckdb.DuckDBPyConnection):
        con.execute(SCHEMA_SQL)
        self._migrate_additive(con)

    def _migrate_additive(self, con: duckdb.DuckDBPyConnection):
        def _add(tbl, col, typ):
            cols = [r[1] for r in con.execute(f"PRAGMA table_info('{tbl}')").fetchall()]
            if col not in cols: 
                con.execute(f"ALTER TABLE {tbl} ADD COLUMN {col} {typ}")

        _add("snapshots", "is_early_close", "BOOLEAN")
        _add("snapshots", "market_close_utc", "TIMESTAMP")
        _add("snapshots", "post_end_utc", "TIMESTAMP")
        _add("snapshots", "seconds_to_close", "INTEGER")
        _add("predictions", "prob_flat", "DOUBLE")
        _add("predictions", "horizon_kind", "TEXT DEFAULT 'FIXED'")
        _add("predictions", "horizon_seconds", "INTEGER")
        _add("predictions", "is_mock", "BOOLEAN DEFAULT FALSE")
        _add("predictions", "meta_json", "JSON")
        _add("endpoint_state", "last_change_ts_utc", "TIMESTAMP")
        _add("endpoint_state", "last_change_event_id", "UUID")
        _add("endpoint_state", "last_attempt_event_id", "UUID")
        _add("endpoint_state", "last_attempt_ts_utc", "TIMESTAMP")
        _add("endpoint_state", "last_attempt_http_status", "INTEGER")
        _add("endpoint_state", "last_attempt_error_type", "TEXT")
        _add("endpoint_state", "last_attempt_error_msg", "TEXT")
        _add("snapshot_lineage", "payload_class", "TEXT")
        _add("snapshot_lineage", "na_reason", "TEXT")
        _add("snapshot_lineage", "meta_json", "JSON")

    # Step 1 Requirement: Efficient payload retrieval by event_id list
    def get_payloads_by_event_ids(self, con: duckdb.DuckDBPyConnection, event_ids: List[str]) -> Dict[str, Any]:
        if not event_ids: return {}
        placeholders = ','.join(['?'] * len(event_ids))
        query = f"SELECT event_id, payload_json FROM raw_http_events WHERE event_id IN ({placeholders})"
        rows = con.execute(query, event_ids).fetchall()
        
        res = {}
        for row in rows:
            eid, pj = str(row[0]), row[1]
            if pj is not None:
                try:
                    res[eid] = json.loads(pj) if isinstance(pj, str) else pj
                except Exception:
                    res[eid] = None
            else:
                res[eid] = None
        return res

    def insert_snapshot(
        self, con, *, run_id, asof_ts_utc, ticker, session_label, is_trading_day, is_early_close: bool, 
        data_quality_score, market_close_utc: Optional[datetime], post_end_utc: Optional[datetime], seconds_to_close: Optional[int]
    ) -> str:
        row = con.execute("SELECT snapshot_id FROM snapshots WHERE ticker=? AND asof_ts_utc=?", [ticker.upper(), asof_ts_utc]).fetchone()
        if row: return str(row[0])
        snapshot_id = uuid.uuid4()
        con.execute(
            """INSERT INTO snapshots (snapshot_id, run_id, asof_ts_utc, ticker, session_label, is_trading_day, is_early_close, data_quality_score, market_close_utc, post_end_utc, seconds_to_close, created_at_utc) 
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""", 
            [str(snapshot_id), str(run_id), asof_ts_utc, ticker.upper(), session_label, is_trading_day, is_early_close, data_quality_score, market_close_utc, post_end_utc, seconds_to_close, datetime.now(UTC)]
        )
        return str(snapshot_id)

    def insert_prediction(self, con, p):
        snapshot_id = str(p["snapshot_id"])
        horizon_kind = p.get("horizon_kind", "FIXED")
        horizon_minutes = p.get("horizon_minutes")
        horizon_seconds = p.get("horizon_seconds")

        if horizon_kind == "TO_CLOSE":
            if horizon_minutes is None: horizon_minutes = 0
            if horizon_seconds is None: raise ValueError("TO_CLOSE prediction requires horizon_seconds")
        elif horizon_minutes is not None:
            horizon_minutes = int(horizon_minutes)

        con.execute(
            "DELETE FROM predictions WHERE snapshot_id=? AND horizon_kind=? AND horizon_minutes IS NOT DISTINCT FROM ? AND horizon_seconds IS NOT DISTINCT FROM ?", 
            [snapshot_id, horizon_kind, horizon_minutes, horizon_seconds]
        )
        pid = uuid.uuid4()
        con.execute(
            """INSERT INTO predictions (prediction_id, snapshot_id, horizon_minutes, horizon_kind, horizon_seconds, start_price, bias, confidence, prob_up, prob_down, prob_flat, model_name, model_version, model_hash, is_mock, meta_json) 
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            [str(pid), snapshot_id, horizon_minutes, horizon_kind, horizon_seconds, p.get("start_price"), p.get("bias"), p["confidence"], p["prob_up"], p["prob_down"], p["prob_flat"], p["model_name"], p["model_version"], p["model_hash"], p.get("is_mock", False), json.dumps(p.get("meta_json", {}))]
        )
        return pid

    def upsert_endpoint(self, con, method, path, params, registry):
        sig = registry.signature(method, path, params)
        row = con.execute("SELECT endpoint_id FROM dim_endpoints WHERE signature=?", [sig]).fetchone()
        if row: return row[0]
        eid = con.execute("SELECT nextval('seq_endpoint_id')").fetchone()[0]
        con.execute("INSERT INTO dim_endpoints (endpoint_id, method, path, signature, params_hash, params_json) VALUES (?,?,?,?,?,?)", [eid, method, path, sig, registry.params_hash(params), json.dumps(params)])
        return eid

    def insert_raw_event(self, con, run_id, ticker, endpoint_id, req_at, rec_at, status, lat, ph, pj, retry, etype, emsg, circ):
        safe_req = to_utc_dt(req_at, fallback=datetime.now(UTC))
        safe_rec = to_utc_dt(rec_at, fallback=safe_req)
        eid = uuid.uuid4()
        con.execute(
            """INSERT INTO raw_http_events (event_id, run_id, requested_at_utc, received_at_utc, ticker, endpoint_id, http_status, latency_ms, payload_hash, payload_json, is_retry, error_type, error_msg, circuit_state_json) 
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", 
            [str(eid), str(run_id), safe_req, safe_rec, ticker, endpoint_id, status, lat, ph, json.dumps(pj) if pj else None, retry, etype, emsg, json.dumps(circ) if circ else None]
        )
        return eid

    def begin_run(self, con, asof_ts_utc, session_label, is_trading_day, is_early_close, config_version, api_catalog_hash, notes=""):
        run_id = uuid.uuid4()
        con.execute(
            """INSERT INTO meta_runs (run_id, started_at_utc, ended_at_utc, asof_ts_utc, session_label, is_trading_day, is_early_close, config_version, api_catalog_hash, notes) 
               VALUES (?,?,?,?,?,?,?,?,?,?)""", 
            [str(run_id), datetime.now(UTC), None, asof_ts_utc, session_label, is_trading_day, is_early_close, config_version, api_catalog_hash, notes]
        )
        return run_id

    def insert_config(self, con, yaml_text):
        h = hashlib.sha256(yaml_text.encode()).hexdigest()
        ver = con.execute("SELECT nextval('seq_config_version')").fetchone()[0]
        con.execute("INSERT INTO meta_config (config_version, config_hash, config_yaml, created_at_utc) VALUES (?,?,?,?)", [ver, h, yaml_text, datetime.now(UTC)])
        con.execute("INSERT INTO config_history (config_version, ingested_at_utc, yaml_content) VALUES (?,?,?)", [str(ver), datetime.now(UTC), yaml_text])
        return ver

    def upsert_tickers(self, con, tickers):
        con.executemany("INSERT OR IGNORE INTO dim_tickers (ticker) VALUES (?)", [[t.upper()] for t in tickers])

    def get_endpoint_state(self, con, ticker: str, endpoint_id: int) -> Optional[Dict[str, Any]]:
        row = con.execute("SELECT last_success_event_id, last_success_ts_utc, last_payload_hash, last_change_ts_utc, last_change_event_id FROM endpoint_state WHERE ticker=? AND endpoint_id=?", [ticker, endpoint_id]).fetchone()
        if not row: return None
        return {
            "last_success_event_id": str(row[0]) if row[0] else None,
            "last_success_ts_utc": row[1],
            "last_payload_hash": row[2],
            "last_change_ts_utc": row[3],
            "last_change_event_id": str(row[4]) if row[4] else None
        }

    def upsert_endpoint_state(self, con, ticker: str, endpoint_id: int, event_id: str, res: Any, attempt_ts_utc: datetime, is_success_class: bool, changed: bool) -> None:
        con.execute("INSERT OR IGNORE INTO endpoint_state (ticker, endpoint_id) VALUES (?,?)", [ticker, endpoint_id])
        con.execute(
            """UPDATE endpoint_state SET last_attempt_event_id=?, last_attempt_ts_utc=?, last_attempt_http_status=?, last_attempt_error_type=?, last_attempt_error_msg=? WHERE ticker=? AND endpoint_id=?""",
            [str(event_id), attempt_ts_utc, res.status_code, res.error_type, res.error_message, ticker, endpoint_id]
        )
        if is_success_class:
            con.execute(
                """UPDATE endpoint_state SET last_success_event_id=?, last_success_ts_utc=?, last_payload_hash=?, last_change_ts_utc=CASE WHEN ? THEN ? ELSE last_change_ts_utc END, last_change_event_id=CASE WHEN ? THEN ? ELSE last_change_event_id END WHERE ticker=? AND endpoint_id=?""",
                [str(event_id), attempt_ts_utc, res.payload_hash, changed, attempt_ts_utc, changed, str(event_id), ticker, endpoint_id]
            )

    # Updated Feature / Level inserts mapped directly to the unified MetaContract lists
    def insert_features(self, con, snapshot_id, features_with_meta: List[Dict[str, Any]]):
        for f in features_with_meta:
            con.execute(
                "INSERT INTO features (snapshot_id, feature_key, feature_value, meta_json) VALUES (?,?,?,?)", 
                [str(snapshot_id), f["feature_key"], f["feature_value"], json.dumps(f["meta_json"])]
            )
            
    def insert_levels(self, con, snapshot_id, levels: List[Dict[str, Any]]):
        if not levels: return
        con.executemany(
            "INSERT INTO derived_levels (snapshot_id, level_type, price, magnitude, meta_json) VALUES (?,?,?,?,?)", 
            [[str(snapshot_id), l["level_type"], l["price"], l["magnitude"], json.dumps(l.get("meta_json", {}))] for l in levels]
        )
        
    def insert_lineage(self, con, snapshot_id, endpoint_id, used_event_id, freshness_state, data_age_seconds, payload_class, na_reason, meta_json):
        con.execute(
            """INSERT INTO snapshot_lineage (snapshot_id, endpoint_id, used_event_id, freshness_state, data_age_seconds, payload_class, na_reason, meta_json) VALUES (?,?,?,?,?,?,?,?)""", 
            [str(snapshot_id), endpoint_id, str(used_event_id) if used_event_id else None, freshness_state, data_age_seconds, payload_class, na_reason, json.dumps(meta_json) if meta_json else None]
        )
            
    def end_run(self, con, run_id):
        con.execute("UPDATE meta_runs SET ended_at_utc=? WHERE run_id=?", [datetime.now(UTC), str(run_id)])
        
    def ro_connect(self) -> duckdb.DuckDBPyConnection: return duckdb.connect(self.duckdb_path, read_only=True)
    @contextlib.contextmanager
    def writer(self):
        con = self._connect_new()
        try: yield con
        finally: con.close()