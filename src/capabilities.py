from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

import duckdb

from .api_catalog_loader import load_api_catalog, EndpointRegistry
from .config_loader import load_endpoint_plan
from .metric_specs import INSTITUTIONAL_METRICS, EndpointRef, MetricSpec

UTC = timezone.utc


# [Fix: Step 2] Make capabilities library-safe.
class CapabilitiesError(Exception):
    """Raised when capabilities checker encounters fatal state or schema errors."""
    pass


# [Fix: Step 6] Typed struct to formalize truth extraction returns.
@dataclass(frozen=True)
class EndpointTruth:
    has_success: bool
    last_status: Optional[int]
    last_error: Optional[str]
    last_success_age_s: Optional[int]
    last_change_age_s: Optional[int]
    observed_keys: List[str]
    selected_endpoint_id: Optional[int]
    selected_success_event_id: Optional[str]


def flatten_keys(obj: Any, prefix: str, list_cap: int, out: Set[str]) -> None:
    """
    Deterministically flatten JSON keys into dot paths.
    Only emits leaf paths (scalars), not container nodes.
    """
    if isinstance(obj, dict):
        for k in sorted(obj.keys()):
            new_prefix = f"{prefix}.{k}" if prefix else k
            # Do not add container key itself
            flatten_keys(obj[k], new_prefix, list_cap, out)
    elif isinstance(obj, list):
        for item in obj[:list_cap]:
            new_prefix = f"{prefix}.[]" if prefix else "[]"
            flatten_keys(item, new_prefix, list_cap, out)
    else:
        # Scalar or None: this is a leaf
        if prefix:
            out.add(prefix)


def ensure_utc(ts: Optional[datetime]) -> Optional[datetime]:
    if ts is None:
        return None
    if ts.tzinfo is None:
        return ts.replace(tzinfo=UTC)
    return ts.astimezone(UTC)


class CapabilitiesChecker:
    def __init__(self, catalog: Any, plan_yaml: Dict[str, Any], db_path: str, validate_db_schema: bool = True):
        self.catalog = catalog
        self.plan_yaml = plan_yaml
        self.db_path = db_path
        
        # [Fix: Step 8] Map endpoints to the tiers they belong to
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

        if validate_db_schema:
            self._validate_db_schema()

    @classmethod
    def from_paths(cls, catalog_path: str, db_path: str, plan_path: str) -> CapabilitiesChecker:
        catalog = load_api_catalog(catalog_path)
        plan_yaml = load_endpoint_plan(plan_path)
        return cls(catalog, plan_yaml, db_path)

    def _validate_db_schema(self) -> None:
        # [Fix: Step 3] Validate specific required columns, not just table presence.
        try:
            with duckdb.connect(self.db_path, read_only=True) as con:
                tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
                required_tables = {"dim_endpoints", "endpoint_state", "raw_http_events"}
                if not required_tables.issubset(tables):
                    raise CapabilitiesError(f"DB not initialized. Missing tables: {required_tables - tables}")
                
                def check_columns(table: str, required_cols: Set[str]):
                    cols = {r[1] for r in con.execute(f"PRAGMA table_info('{table}')").fetchall()}
                    missing = required_cols - cols
                    if missing:
                        raise CapabilitiesError(f"DB schema mismatch: table '{table}' missing required columns {missing}")

                check_columns("dim_endpoints", {"endpoint_id", "method", "path"})
                check_columns("endpoint_state", {"ticker", "endpoint_id", "last_success_event_id", "last_success_ts_utc", "last_change_ts_utc"})
                check_columns("raw_http_events", {"event_id", "endpoint_id", "received_at_utc", "http_status", "error_type", "payload_json"})

        except duckdb.Error as e:
            raise CapabilitiesError(f"Failed to connect to DuckDB at {self.db_path}: {e}")

    def extract_db_truth(self, con: duckdb.DuckDBPyConnection, method: str, path: str) -> EndpointTruth:
        now_utc = datetime.now(UTC)
        
        # [Fix: Step 5] Resolve the EXACT "winner" variant across multiple endpoint_id permutations.
        variant_row = con.execute("""
            WITH ep_max AS (
                SELECT endpoint_id,
                       MAX(last_success_ts_utc) AS max_succ,
                       MAX(last_change_ts_utc) AS max_chg
                FROM endpoint_state
                GROUP BY endpoint_id
            )
            SELECT d.endpoint_id
            FROM dim_endpoints d
            LEFT JOIN ep_max e ON d.endpoint_id = e.endpoint_id
            WHERE d.method = ? AND d.path = ?
            ORDER BY COALESCE(e.max_succ, e.max_chg) DESC NULLS LAST, d.endpoint_id DESC
            LIMIT 1
        """, [method.upper(), path]).fetchone()

        if not variant_row:
            return EndpointTruth(False, None, None, None, None, [], None, None)

        winner_eid = variant_row[0]

        # [Fix: Step 4] Deterministic best success across ALL tickers for the winner variant
        best_success_row = con.execute("""
            SELECT last_success_event_id, last_success_ts_utc
            FROM endpoint_state
            WHERE endpoint_id = ? AND last_success_event_id IS NOT NULL
            ORDER BY last_success_ts_utc DESC NULLS LAST, ticker ASC
            LIMIT 1
        """, [winner_eid]).fetchone()

        # [Fix: Step 4] Deterministic latest attempt across ALL tickers
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
        selected_event_id: Optional[str] = None
        
        latest_status = None
        latest_error = None
        latest_attempt_ts: Optional[datetime] = None

        if best_success_row:
            evt_id, succ_ts = best_success_row
            best_success_ts = ensure_utc(succ_ts)
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

        last_success_age = int((now_utc - best_success_ts).total_seconds()) if best_success_ts else None
        last_change_age = int((now_utc - latest_attempt_ts).total_seconds()) if latest_attempt_ts else None

        return EndpointTruth(
            has_success=found_success,
            last_status=latest_status,
            last_error=latest_error,
            last_success_age_s=last_success_age,
            last_change_age_s=last_change_age,
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
            
            # [Fix: Step 7] Explicit strictness evaluation
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
                    "last_change_age_s": state.last_change_age_s,
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
                        age_str = f"{state.last_change_age_s}s ago" if state.last_change_age_s is not None else "N/A"
                        print(f"  -> N/A: Never seen successful payload. Last attempt: {state.last_status} ({state.last_error or 'no error'}) {age_str}")
                    else:
                        print("  -> N/A: Never seen successful payload (no attempts found)")
                else:
                    sa = f"{state.last_success_age_s}s ago" if state.last_success_age_s is not None else "N/A"
                    ca = f"{state.last_change_age_s}s ago" if state.last_change_age_s is not None else "N/A"
                    print(f"  Last Success: {sa} | Last Change: {ca} | Status: {state.last_status}")
                    
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