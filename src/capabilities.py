from __future__ import annotations
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple
import duckdb

from .api_catalog_loader import load_api_catalog
from .config_loader import load_endpoint_plan
from .metric_specs import INSTITUTIONAL_METRICS, EndpointRef, MetricSpec

UTC = timezone.utc

class CapabilitiesError(Exception): pass
class DbOpenError(CapabilitiesError): pass
class SchemaMismatchError(CapabilitiesError): pass

@dataclass(frozen=True)
class EndpointTruth:
    has_success: bool
    last_status: Optional[int]
    last_error: Optional[str]
    last_success_age_s: Optional[int]
    last_payload_change_age_s: Optional[int]
    last_attempt_age_s: Optional[int]
    observed_keys: List[str]
    selected_endpoint_id: Optional[int]
    selected_success_event_id: Optional[str]

def flatten_keys(obj: Any, prefix: str, list_cap: int, out: Set[str]) -> None:
    if isinstance(obj, dict):
        for k in sorted(obj.keys()):
            flatten_keys(obj[k], f"{prefix}.{k}" if prefix else k, list_cap, out)
    elif isinstance(obj, list):
        for item in obj[:list_cap]:
            flatten_keys(item, f"{prefix}.[]" if prefix else "[]", list_cap, out)
    else:
        if prefix: out.add(prefix)

def ensure_utc(ts: Optional[datetime]) -> Optional[datetime]:
    if ts is None: return None
    if ts.tzinfo is None: return ts.replace(tzinfo=UTC)
    return ts.astimezone(UTC)

class CapabilitiesChecker:
    def __init__(self, catalog: Any, plan_yaml: Dict[str, Any], db_path: str, validate_schema: bool = True):
        self.catalog = catalog
        self.plan_yaml = plan_yaml
        self.db_path = db_path
        
        self.planned_endpoints: Dict[Tuple[str, str], List[str]] = {}
        plans = self.plan_yaml.get("plans", {})
        for tier_name, entries in plans.items():
            for entry in entries:
                key = (str(entry.get("method", "")).upper(), str(entry.get("path", "")))
                if key not in self.planned_endpoints: self.planned_endpoints[key] = []
                self.planned_endpoints[key].append(tier_name)

        if validate_schema: self._validate_db_schema()

    @classmethod
    def from_paths(cls, catalog_path: str, db_path: str, plan_path: str) -> CapabilitiesChecker:
        return cls(load_api_catalog(catalog_path), load_endpoint_plan(plan_path), db_path)

    def _validate_db_schema(self) -> None:
        try:
            with duckdb.connect(self.db_path, read_only=True) as con:
                tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
                required_tables = {"dim_endpoints", "endpoint_state", "raw_http_events"}
                if not required_tables.issubset(tables): raise SchemaMismatchError(f"Missing tables: {required_tables - tables}")
                def check_columns(table: str, required_cols: Set[str]):
                    cols = {r[1] for r in con.execute(f"PRAGMA table_info('{table}')").fetchall()}
                    missing = required_cols - cols
                    if missing: raise SchemaMismatchError(f"Table '{table}' missing required columns {missing}")
                check_columns("dim_endpoints", {"endpoint_id", "method", "path"})
                check_columns("endpoint_state", {"ticker", "endpoint_id", "last_success_event_id", "last_success_ts_utc", "last_change_ts_utc"})
                check_columns("raw_http_events", {"event_id", "endpoint_id", "received_at_utc", "http_status", "error_type", "payload_json"})
        except duckdb.Error as e: raise DbOpenError(f"Failed to connect to DuckDB: {e}")

    def extract_db_truth(self, con: duckdb.DuckDBPyConnection, method: str, path: str) -> EndpointTruth:
        now_utc = datetime.now(UTC)
        
        ticker_cond = "AND es.ticker = '__MARKET__'" if path.startswith("/api/market/") else ""
        ticker_rh = "AND rh.ticker = '__MARKET__'" if path.startswith("/api/market/") else ""
        ticker_base = "AND ticker = '__MARKET__'" if path.startswith("/api/market/") else ""
        
        variant_row = con.execute(f"""
            WITH ep_max AS (
                SELECT e.endpoint_id,
                       MAX(es.last_success_ts_utc) AS max_succ,
                       MAX(rh.received_at_utc) AS max_attempt
                FROM dim_endpoints e
                LEFT JOIN endpoint_state es ON e.endpoint_id = es.endpoint_id {ticker_cond}
                LEFT JOIN raw_http_events rh ON e.endpoint_id = rh.endpoint_id {ticker_rh}
                WHERE e.method = ? AND e.path = ?
                GROUP BY e.endpoint_id
            )
            SELECT endpoint_id FROM ep_max ORDER BY COALESCE(max_succ, max_attempt) DESC NULLS LAST, endpoint_id DESC LIMIT 1
        """, [method.upper(), path]).fetchone()

        if not variant_row: return EndpointTruth(False, None, None, None, None, None, [], None, None)
        winner_eid = variant_row[0]

        best_success_row = con.execute(f"""
            SELECT last_success_event_id, last_success_ts_utc, last_change_ts_utc
            FROM endpoint_state WHERE endpoint_id = ? AND last_success_event_id IS NOT NULL {ticker_base}
            ORDER BY last_success_ts_utc DESC NULLS LAST, ticker ASC LIMIT 1
        """, [winner_eid]).fetchone()

        latest_attempt_row = con.execute(f"""
            SELECT http_status, error_type, received_at_utc
            FROM raw_http_events WHERE endpoint_id = ? {ticker_base}
            ORDER BY received_at_utc DESC NULLS LAST, event_id ASC LIMIT 1
        """, [winner_eid]).fetchone()

        found_success, observed_keys = False, set()
        best_success_ts, best_change_ts, selected_event_id = None, None, None
        latest_status, latest_error, latest_attempt_ts = None, None, None

        if best_success_row:
            evt_id, succ_ts, chg_ts = best_success_row
            best_success_ts, best_change_ts, selected_event_id = ensure_utc(succ_ts), ensure_utc(chg_ts), str(evt_id)
            payload_row = con.execute("SELECT payload_json FROM raw_http_events WHERE event_id = ?", [evt_id]).fetchone()
            if payload_row and payload_row[0]:
                found_success = True
                try: flatten_keys(json.loads(payload_row[0]) if isinstance(payload_row[0], str) else payload_row[0], "", 5, observed_keys)
                except Exception: pass

        if not found_success and winner_eid:
            fallback_row = con.execute(f"""
                SELECT event_id, payload_json FROM raw_http_events 
                WHERE endpoint_id = ? AND http_status BETWEEN 200 AND 299 AND payload_json IS NOT NULL {ticker_base}
                ORDER BY received_at_utc DESC LIMIT 1
            """, [winner_eid]).fetchone()
            if fallback_row:
                found_success = True
                try: flatten_keys(json.loads(fallback_row[1]) if isinstance(fallback_row[1], str) else fallback_row[1], "", 5, observed_keys)
                except Exception: pass

        if latest_attempt_row:
            latest_status, latest_error, rec_ts = latest_attempt_row
            latest_attempt_ts = ensure_utc(rec_ts)

        return EndpointTruth(
            has_success=found_success, last_status=latest_status, last_error=latest_error,
            last_success_age_s=int((now_utc - best_success_ts).total_seconds()) if best_success_ts else None,
            last_payload_change_age_s=int((now_utc - best_change_ts).total_seconds()) if best_change_ts else None,
            last_attempt_age_s=int((now_utc - latest_attempt_ts).total_seconds()) if latest_attempt_ts else None,
            observed_keys=sorted(list(observed_keys)), selected_endpoint_id=winner_eid, selected_success_event_id=selected_event_id
        )

    def evaluate_metric(self, spec: MetricSpec, db_states: Dict[EndpointRef, EndpointTruth]) -> Dict[str, Any]:
        missing_endpoints, missing_keys_reasons, used_presence_only = [], [], False
        for ep in spec.required_endpoints:
            if not self.catalog.has(ep.method, ep.path):
                missing_endpoints.append(f"{ep.method} {ep.path} (not in catalog)"); continue
            state = db_states.get(ep)
            if not state or not state.has_success:
                missing_endpoints.append(f"{ep.method} {ep.path} (never seen successful payload)"); continue
            rule = spec.required_keys_by_endpoint.get(ep)
            if not rule:
                if ep in spec.presence_only_endpoints: used_presence_only = True
                else: missing_keys_reasons.append(f"{ep.method} {ep.path} (no key rules defined for required endpoint)")
                continue
            obs_keys = set(state.observed_keys)
            if rule.all_of and not rule.all_of.issubset(obs_keys): missing_keys_reasons.append(f"{ep.method} {ep.path} missing required keys: {sorted(rule.all_of - obs_keys)}")
            if rule.any_of and not any(alt.issubset(obs_keys) for alt in rule.any_of): missing_keys_reasons.append(f"{ep.method} {ep.path} failed all `any_of` alternative keysets")

        status, reasons = "COMPUTABLE", missing_endpoints + missing_keys_reasons
        if reasons: status = "N/A"
        elif used_presence_only: status = "COMPUTABLE_PRESENCE_ONLY"
        return {"name": spec.name, "status": status, "reasons": reasons}

    def run(self, output_format: str) -> None:
        critical_eps = {ep for m in INSTITUTIONAL_METRICS for ep in m.required_endpoints}
        sorted_eps = sorted(list(critical_eps), key=lambda x: (x.method, x.path))
        
        with duckdb.connect(self.db_path, read_only=True) as con:
            db_states = {ep: self.extract_db_truth(con, ep.method, ep.path) for ep in sorted_eps}

        metric_results = [self.evaluate_metric(m, db_states) for m in INSTITUTIONAL_METRICS]

        if output_format == "json":
            out = {
                "meta": {"catalog_hash": self.catalog.catalog_hash, "db_path": self.db_path, "generated_at_utc": datetime.now(UTC).isoformat()},
                "endpoints": [{
                    "method": ep.method, "path": ep.path,
                    "operation_id": self.catalog.get(ep.method, ep.path).operation_id if self.catalog.has(ep.method, ep.path) else "UNKNOWN",
                    "in_plan": (ep.method.upper(), ep.path) in self.planned_endpoints,
                    "plan_tiers": self.planned_endpoints.get((ep.method.upper(), ep.path), []),
                    **{k: getattr(db_states[ep], k) for k in ["has_success", "last_status", "last_error", "last_success_age_s", "last_payload_change_age_s", "last_attempt_age_s", "selected_endpoint_id", "selected_success_event_id"]},
                    "observed_keys_count": len(db_states[ep].observed_keys),
                    "observed_keys_preview": db_states[ep].observed_keys[:20]
                } for ep in sorted_eps],
                "metrics": metric_results
            }
            print(json.dumps(out, indent=2))
        else:
            print("=" * 80 + f"\nCAPABILITIES REPORT\nCatalog Hash: {self.catalog.catalog_hash}\nDB Path: {self.db_path}\nGenerated At: {datetime.now(UTC).isoformat()}\n" + "=" * 80 + "\nENDPOINT STATUS (Critical)\n" + "-" * 80)
            for ep in sorted_eps:
                tiers = self.planned_endpoints.get((ep.method.upper(), ep.path), [])
                state, op_id = db_states[ep], (self.catalog.get(ep.method, ep.path).operation_id if self.catalog.has(ep.method, ep.path) else "UNKNOWN")
                print(f"{ep.method} {ep.path} [opId: {op_id}] (in_plan: True (tiers: {tiers}) if tiers else 'in_plan: False')")
                if not state.has_success:
                    print(f"  -> N/A: Never seen successful payload. Last attempt: {state.last_status} ({state.last_error or 'no error'}) {state.last_attempt_age_s}s ago" if state.last_status or state.last_error else "  -> N/A: Never seen successful payload (no attempts found)")
                else:
                    print(f"  Last Success: {state.last_success_age_s}s ago | Last Payload Change: {state.last_payload_change_age_s}s ago | Last Attempt Status: {state.last_status}\n  Keys ({len(state.observed_keys)} observed): {state.observed_keys[:10]} {'...' if len(state.observed_keys) > 10 else ''}")
                print()
            print("=" * 80 + "\nMETRIC COMPUTABILITY\n" + "-" * 80)
            for mr in metric_results:
                print(f"[{mr['status']}]".ljust(26) + f" {mr['name']}")
                if "COMPUTABLE" in mr['status']: print("  All required endpoints and keys are present.")
                else:
                    for r in mr["reasons"]: print(f"  Reason: {r}")
                print()

def check_capabilities(catalog_path: str, db_path: str, plan_path: str, output_format: str = "text") -> None:
    CapabilitiesChecker.from_paths(catalog_path, db_path, plan_path).run(output_format)