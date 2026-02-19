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


# [Fix: Step 2] Make capabilities library-safe with Typed Exceptions.
class CapabilitiesError(Exception):
    """Base exception for Capabilities logic."""
    pass

class DbOpenError(CapabilitiesError):
    """Raised when the database cannot be opened."""
    pass

class SchemaMismatchError(CapabilitiesError):
    """Raised when the database lacks required tables or columns."""
    pass


# [Fix: Step 4 & 6] Typed struct formalizing distinct timestamp semantics.
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
    """Deterministically flatten JSON keys into dot paths (emits leaf nodes)."""
    if isinstance(obj, dict):
        for k in sorted(obj.keys()):
            new_prefix = f"{prefix}.{k}" if prefix else k
            flatten_keys(obj[k], new_prefix, list_cap, out)
    elif isinstance(obj, list):
        for item in obj[:list_cap]:
            new_prefix = f"{prefix}.[]" if prefix else "[]"
            flatten_keys(item, new_prefix, list_cap, out)
    else:
        if prefix:
            out.add(prefix)


def ensure_utc(ts: Optional[datetime]) -> Optional[datetime]:
    if ts is None:
        return None
    if ts.tzinfo is None:
        return ts.replace(tzinfo=UTC)
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
                m = str(entry.get("method", "")).upper()
                p = str(entry.get("path", ""))
                key = (m, p)
                if key not in self.planned_endpoints:
                    self.planned_endpoints[key] = []
                self.planned_endpoints[key].append(tier_name)

        if validate_schema:
            self._validate_db_schema()

    @classmethod
    def from_paths(cls, catalog_path: str, db_path: str, plan_path: str) -> CapabilitiesChecker:
        catalog = load_api_catalog(catalog_path)
        plan_yaml = load_endpoint_plan(plan_path)
        return cls(catalog, plan_yaml, db_path)

    def _validate_db_schema(self) -> None:
        # [Fix: Step 3] Validate specific required columns, preventing schema drift silently.
        try:
            with duckdb.connect(self.db_path, read_only=True) as con:
                tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
                required_tables = {"dim_endpoints", "endpoint_state", "raw_http_events"}
                if not required_tables.issubset(tables):
                    raise SchemaMismatchError(f"Missing tables: {required_tables - tables}")
                
                def check_columns(table: str, required_cols: Set[str]):
                    cols = {r[1] for r in con.execute(f"PRAGMA table_info('{table}')").fetchall()}
                    missing = required_cols - cols
                    if missing:
                        raise SchemaMismatchError(f"Table '{table}' missing required columns {missing}")

                check_columns("dim_endpoints", {"endpoint_id", "method", "path"})
                check_columns("endpoint_state", {"ticker", "endpoint_id", "last_success_event_id", "last_success_ts_utc", "last_change_ts_utc"})
                check_columns("raw_http_events", {"event_id", "endpoint_id", "received_at_utc", "http_status", "error_type", "payload_json"})

        except duckdb.Error as e:
            raise DbOpenError(f"Failed to connect to DuckDB at {self.db_path}: {e}")

    def extract_db_truth(self, con: duckdb.DuckDBPyConnection, method: str, path: str) -> EndpointTruth:
        now_utc = datetime.now(UTC)
        
        # [Fix: Step 5] Rank variants deterministically even for "failures-only" endpoints
        variant_row = con.execute("""
            WITH ep_max AS (
                SELECT e.endpoint_id,
                       MAX(es.last_success_ts_utc) AS max_succ,
                       MAX(rh.received_at_utc) AS max_attempt
                FROM dim_endpoints e
                LEFT JOIN endpoint_state es ON e.endpoint_id = es.endpoint_id
                LEFT JOIN raw_http_events rh ON e.endpoint_id = rh.endpoint_id
                WHERE e.method = ? AND e.path = ?
                GROUP BY e.endpoint_id
            )
            SELECT endpoint_id
            FROM ep_max
            ORDER BY COALESCE(max_succ, max_attempt) DESC NULLS LAST, endpoint_id DESC
            LIMIT 1
        """, [method.upper(), path]).fetchone()

        if not variant_row:
            return EndpointTruth(False, None, None, None, None, None, [], None, None)

        winner_eid = variant_row[0]

        # [Fix: Step 4] Fetch 'best success' mapping exactly to multi-ticker guarantees
        best_success_row = con.execute("""
            SELECT last_success_event_id, last_success_ts_utc, last_change_ts_utc
            FROM endpoint_state
            WHERE endpoint_id = ? AND last_success_event_id IS NOT NULL
            ORDER BY last_success_ts_utc DESC NULLS LAST, ticker ASC
            LIMIT 1
        """, [winner_eid]).fetchone()

        # [Fix: Step 4] Fetch 'latest attempt' mapping
        latest_attempt_row = con.execute("""
            SELECT http_status, error_type, received_at_utc
            FROM raw_http_events
            WHERE endpoint_id = ?
            ORDER BY received_at_utc DESC NULLS LAST, event_id ASC
            LIMIT 1
        """, [winner_eid]).fetchone()

        found_success = False
        observed_keys: Set[str] = set()
        
        best_success_ts: Optional[datetime] = None
        best_change_ts: Optional[datetime] = None
        selected_event_id: Optional[str] = None
        
        latest_status = None
        latest_error = None
        latest_attempt_ts: Optional[datetime] = None

        if best_success_row:
            evt_id, succ_ts, chg_ts = best_success_row
            best_success_ts = ensure_utc(succ_ts)
            best_change_ts = ensure_utc(chg_ts)
            selected_event_id = str(evt_id)
            
            payload_row = con.execute(
                "SELECT payload_json FROM raw_http_events WHERE event_id = ?",
                [evt_id]
            ).fetchone()
            
            if payload_row and payload_row[0]:
                found_success = True
                try:
                    pj = json.loads(payload_row[0]) if isinstance(payload_row[0], str) else payload_row[0]
                    flatten_keys(pj, "", 5, observed_keys)
                except Exception:
                    pass

        if latest_attempt_row:
            status, error, rec_ts = latest_attempt_row
            latest_status = status
            latest_error = error
            latest_attempt_ts = ensure_utc(rec_ts)

        # Disambiguated time aggregations
        last_success_age = int((now_utc - best_success_ts).total_seconds()) if best_success_ts else None
        last_payload_change_age = int((now_utc - best_change_ts).total_seconds()) if best_change_ts else None
        last_attempt_age = int((now_utc - latest_attempt_ts).total_seconds()) if latest_attempt_ts else None

        return EndpointTruth(
            has_success=found_success,
            last_status=latest_status,
            last_error=latest_error,
            last_success_age_s=last_success_age,
            last_payload_change_age_s=last_payload_change_age,
            last_attempt_age_s=last_attempt_age,
            observed_keys=sorted(list(observed_keys)),
            selected_endpoint_id=winner_eid,
            selected_success_event_id=selected_event_id
        )

    def evaluate_metric(self, spec: MetricSpec, db_states: Dict[EndpointRef, EndpointTruth]) -> Dict[str, Any]:
        missing_endpoints = []
        missing_keys_reasons = []
        used_presence_only = False

        for ep in spec.required_endpoints:
            if not self.catalog.has(ep.method, ep.path):
                missing_endpoints.append(f"{ep.method} {ep.path} (not in catalog)")
                continue

            state = db_states.get(ep)
            if not state or not state.has_success:
                missing_endpoints.append(f"{ep.method} {ep.path} (never seen successful payload)")
                continue

            rule = spec.required_keys_by_endpoint.get(ep)
            
            if not rule:
                if ep in spec.presence_only_endpoints:
                    used_presence_only = True
                else:
                    missing_keys_reasons.append(f"{ep.method} {ep.path} (no key rules defined for required endpoint)")
                continue

            obs_keys = set(state.observed_keys)
            if rule.all_of and not rule.all_of.issubset(obs_keys):
                missing = rule.all_of - obs_keys
                missing_keys_reasons.append(f"{ep.method} {ep.path} missing required keys: {sorted(missing)}")

            if rule.any_of:
                passed_any = any(alt.issubset(obs_keys) for alt in rule.any_of)
                if not passed_any:
                    missing_keys_reasons.append(f"{ep.method} {ep.path} failed all `any_of` alternative keysets")

        status = "COMPUTABLE"
        reasons = []
        if missing_endpoints:
            status = "N/A"
            reasons.extend(missing_endpoints)
        if missing_keys_reasons:
            status = "N/A"
            reasons.extend(missing_keys_reasons)

        if status == "COMPUTABLE" and used_presence_only:
            status = "COMPUTABLE_PRESENCE_ONLY"

        return {"name": spec.name, "status": status, "reasons": reasons}

    def run(self, output_format: str) -> None:
        critical_eps: Set[EndpointRef] = set()
        for m in INSTITUTIONAL_METRICS:
            critical_eps.update(m.required_endpoints)

        sorted_eps = sorted(list(critical_eps), key=lambda x: (x.method, x.path))
        
        db_states: Dict[EndpointRef, EndpointTruth] = {}
        with duckdb.connect(self.db_path, read_only=True) as con:
            for ep in sorted_eps:
                db_states[ep] = self.extract_db_truth(con, ep.method, ep.path)

        metric_results = []
        for m in INSTITUTIONAL_METRICS:
            metric_results.append(self.evaluate_metric(m, db_states))

        if output_format == "json":
            endpoint_details = []
            for ep in sorted_eps:
                state = db_states[ep]
                key = (ep.method.upper(), ep.path)
                in_plan = key in self.planned_endpoints
                tiers = self.planned_endpoints.get(key, [])
                op_id = self.catalog.get(ep.method, ep.path).operation_id if self.catalog.has(ep.method, ep.path) else "UNKNOWN"
                
                endpoint_details.append({
                    "method": ep.method,
                    "path": ep.path,
                    "operation_id": op_id,
                    "in_plan": in_plan,
                    "plan_tiers": tiers,
                    "has_success": state.has_success,
                    "last_status": state.last_status,
                    "last_error": state.last_error,
                    "last_success_age_s": state.last_success_age_s,
                    "last_payload_change_age_s": state.last_payload_change_age_s,
                    "last_attempt_age_s": state.last_attempt_age_s,
                    "observed_keys_count": len(state.observed_keys),
                    "observed_keys_preview": state.observed_keys[:20],
                    "selected_endpoint_id": state.selected_endpoint_id,
                    "selected_success_event_id": state.selected_success_event_id
                })

            out = {
                "meta": {
                    "catalog_hash": self.catalog.catalog_hash,
                    "db_path": self.db_path,
                    "generated_at_utc": datetime.now(UTC).isoformat()
                },
                "endpoints": endpoint_details,
                "metrics": metric_results
            }
            print(json.dumps(out, indent=2))
        else:
            print("=" * 80)
            print("CAPABILITIES REPORT")
            print(f"Catalog Hash: {self.catalog.catalog_hash}")
            print(f"DB Path: {self.db_path}")
            print(f"Generated At: {datetime.now(UTC).isoformat()}")
            print("=" * 80)
            print("ENDPOINT STATUS (Critical)")
            print("-" * 80)
            for ep in sorted_eps:
                key = (ep.method.upper(), ep.path)
                tiers = self.planned_endpoints.get(key, [])
                plan_str = f"in_plan: True (tiers: {tiers})" if tiers else "in_plan: False"
                state = db_states[ep]
                op_id = self.catalog.get(ep.method, ep.path).operation_id if self.catalog.has(ep.method, ep.path) else "UNKNOWN"
                
                print(f"{ep.method} {ep.path} [opId: {op_id}] ({plan_str})")
                if not state.has_success:
                    if state.last_status or state.last_error:
                        age_str = f"{state.last_attempt_age_s}s ago" if state.last_attempt_age_s is not None else "N/A"
                        print(f"  -> N/A: Never seen successful payload. Last attempt: {state.last_status} ({state.last_error or 'no error'}) {age_str}")
                    else:
                        print("  -> N/A: Never seen successful payload (no attempts found)")
                else:
                    sa = f"{state.last_success_age_s}s ago" if state.last_success_age_s is not None else "N/A"
                    ca = f"{state.last_payload_change_age_s}s ago" if state.last_payload_change_age_s is not None else "N/A"
                    print(f"  Last Success: {sa} | Last Payload Change: {ca} | Last Attempt Status: {state.last_status}")
                    
                    keys = state.observed_keys
                    preview = keys[:10]
                    suffix = "..." if len(keys) > 10 else ""
                    print(f"  Keys ({len(keys)} observed): {preview} {suffix}")
                print()

            print("=" * 80)
            print("METRIC COMPUTABILITY")
            print("-" * 80)
            for mr in metric_results:
                status_str = f"[{mr['status']}]".ljust(26)
                print(f"{status_str} {mr['name']}")
                if "COMPUTABLE" in mr['status']:
                    print("  All required endpoints and keys are present.")
                else:
                    for reason in mr["reasons"]:
                        print(f"  Reason: {reason}")
                print()

def check_capabilities(catalog_path: str, db_path: str, plan_path: str, output_format: str = "text") -> None:
    checker = CapabilitiesChecker.from_paths(catalog_path, db_path, plan_path)
    checker.run(output_format)