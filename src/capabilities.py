from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

import duckdb

from .api_catalog_loader import load_api_catalog, EndpointRegistry
from .config_loader import load_endpoint_plan
from .metric_specs import INSTITUTIONAL_METRICS, EndpointRef, MetricSpec

UTC = timezone.utc


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
    def __init__(self, catalog: Any, plan_yaml: Dict[str, Any], db_path: str):
        self.catalog = catalog
        self.plan_yaml = plan_yaml
        self.db_path = db_path
        
        # Build set of all planned endpoints from all tiers
        self.planned_endpoints: Set[Tuple[str, str]] = set()
        plans = self.plan_yaml.get("plans", {})
        for tier_name, entries in plans.items():
            for entry in entries:
                m = str(entry.get("method", "")).upper()
                p = str(entry.get("path", ""))
                self.planned_endpoints.add((m, p))

        self._validate_db_schema()

    @classmethod
    def from_paths(cls, catalog_path: str, db_path: str, plan_path: str) -> CapabilitiesChecker:
        catalog = load_api_catalog(catalog_path)
        plan_yaml = load_endpoint_plan(plan_path)
        return cls(catalog, plan_yaml, db_path)

    def _validate_db_schema(self) -> None:
        try:
            with duckdb.connect(self.db_path, read_only=True) as con:
                tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
                required = {"dim_endpoints", "endpoint_state", "raw_http_events"}
                if not required.issubset(set(tables)):
                    print(f"DB not initialized. Missing tables: {required - set(tables)}")
                    sys.exit(1)
        except duckdb.Error as e:
            print(f"Failed to connect to DuckDB at {self.db_path}: {e}")
            sys.exit(1)

    def extract_db_truth(self, con: duckdb.DuckDBPyConnection, method: str, path: str) -> Dict[str, Any]:
        """
        Fetch ground truth for an endpoint.
        Prioritizes variants with recent success.
        Reports status/error from the most recent attempt (even if failure).
        """
        now_utc = datetime.now(UTC)
        
        # 1. Select top 3 variants by recency of success (or change if no success)
        variant_rows = con.execute("""
            SELECT t1.endpoint_id 
            FROM dim_endpoints t1
            LEFT JOIN endpoint_state t2 ON t1.endpoint_id = t2.endpoint_id
            WHERE t1.method = ? AND t1.path = ?
            ORDER BY COALESCE(t2.last_success_ts_utc, t2.last_change_ts_utc) DESC NULLS LAST
            LIMIT 3
        """, [method.upper(), path]).fetchall()

        observed_keys: Set[str] = set()
        
        # Track best success (for keys/computability)
        best_success_ts: Optional[datetime] = None
        
        # Track latest attempt (for status/error visibility)
        latest_attempt_ts: Optional[datetime] = None
        latest_status = None
        latest_error = None
        
        found_success = False

        for (eid,) in variant_rows:
            # A. Check for success in endpoint_state
            state_row = con.execute("""
                SELECT last_success_event_id, last_success_ts_utc
                FROM endpoint_state
                WHERE endpoint_id = ? AND last_success_event_id IS NOT NULL
            """, [eid]).fetchone()

            if state_row:
                evt_id, succ_ts = state_row
                succ_ts = ensure_utc(succ_ts)
                
                if succ_ts:
                    # Update best success if this one is newer
                    if best_success_ts is None or succ_ts > best_success_ts:
                        best_success_ts = succ_ts
                        
                        # Fetch payload for keys
                        payload_row = con.execute(
                            "SELECT payload_json FROM raw_http_events WHERE event_id = ?",
                            [evt_id]
                        ).fetchone()
                        
                        if payload_row and payload_row[0]:
                            found_success = True
                            try:
                                pj = json.loads(payload_row[0]) if isinstance(payload_row[0], str) else payload_row[0]
                                # Reset keys for the better variant
                                observed_keys = set()
                                flatten_keys(pj, "", 5, observed_keys)
                            except Exception:
                                pass

            # B. Check for latest raw event (success or failure) to get current status
            latest_evt_row = con.execute("""
                SELECT http_status, error_type, received_at_utc
                FROM raw_http_events
                WHERE endpoint_id = ?
                ORDER BY received_at_utc DESC NULLS LAST
                LIMIT 1
            """, [eid]).fetchone()

            if latest_evt_row:
                status, error, rec_ts = latest_evt_row
                rec_ts = ensure_utc(rec_ts)
                
                if rec_ts:
                    if latest_attempt_ts is None or rec_ts > latest_attempt_ts:
                        latest_attempt_ts = rec_ts
                        latest_status = status
                        latest_error = error

        # Calculate ages
        last_success_age = None
        if best_success_ts:
            last_success_age = int((now_utc - best_success_ts).total_seconds())

        last_change_age = None
        if latest_attempt_ts:
            last_change_age = int((now_utc - latest_attempt_ts).total_seconds())

        return {
            "has_success": found_success,
            "last_status": latest_status,
            "last_error": latest_error,
            "last_success_age": last_success_age,
            "last_change_age": last_change_age,
            "observed_keys": sorted(list(observed_keys))
        }

    def evaluate_metric(self, spec: MetricSpec, db_states: Dict[EndpointRef, Dict[str, Any]]) -> Dict[str, Any]:
        missing_endpoints = []
        missing_keys_reasons = []
        
        # Handle optional presence-only escape hatch if defined on spec
        presence_only = getattr(spec, "presence_only_endpoints", set())

        for ep in spec.required_endpoints:
            if not self.catalog.has(ep.method, ep.path):
                missing_endpoints.append(f"{ep.method} {ep.path} (not in catalog)")
                continue

            state = db_states.get(ep)
            if not state or not state["has_success"]:
                missing_endpoints.append(f"{ep.method} {ep.path} (never seen successful payload)")
                continue

            rule = spec.required_keys_by_endpoint.get(ep)
            
            # STRICT validation: If no rule and not explicitly presence-only, fail.
            if not rule:
                if ep not in presence_only:
                    missing_keys_reasons.append(f"{ep.method} {ep.path} (no key rules defined for required endpoint)")
                continue

            obs_keys = set(state["observed_keys"])
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

        return {"name": spec.name, "status": status, "reasons": reasons}

    def run(self, output_format: str) -> None:
        # 1. Identify Critical Endpoints from Metrics
        critical_eps: Set[EndpointRef] = set()
        for m in INSTITUTIONAL_METRICS:
            critical_eps.update(m.required_endpoints)

        sorted_eps = sorted(list(critical_eps), key=lambda x: (x.method, x.path))
        
        db_states: Dict[EndpointRef, Dict[str, Any]] = {}
        with duckdb.connect(self.db_path, read_only=True) as con:
            for ep in sorted_eps:
                db_states[ep] = self.extract_db_truth(con, ep.method, ep.path)

        # 2. Evaluate Metrics
        metric_results = []
        for m in INSTITUTIONAL_METRICS:
            metric_results.append(self.evaluate_metric(m, db_states))

        # 3. Print Desk-Grade Output
        if output_format == "json":
            # Build endpoint details list for JSON
            endpoint_details = []
            for ep in sorted_eps:
                state = db_states[ep]
                in_plan = (ep.method.upper(), ep.path) in self.planned_endpoints
                op_id = self.catalog.get(ep.method, ep.path).operation_id if self.catalog.has(ep.method, ep.path) else "UNKNOWN"
                endpoint_details.append({
                    "method": ep.method,
                    "path": ep.path,
                    "operation_id": op_id,
                    "in_plan": in_plan,
                    "has_success": state["has_success"],
                    "last_status": state["last_status"],
                    "last_error": state["last_error"],
                    "last_success_age": state["last_success_age"],
                    "last_change_age": state["last_change_age"],
                    "observed_keys_count": len(state["observed_keys"]),
                    "observed_keys_preview": state["observed_keys"][:20]
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
                in_plan = (ep.method.upper(), ep.path) in self.planned_endpoints
                state = db_states[ep]
                op_id = self.catalog.get(ep.method, ep.path).operation_id if self.catalog.has(ep.method, ep.path) else "UNKNOWN"
                
                print(f"{ep.method} {ep.path} [opId: {op_id}] (in_plan: {in_plan})")
                if not state["has_success"]:
                    # Show failure status if available
                    if state["last_status"] or state["last_error"]:
                        age_str = f"{state['last_change_age']}s ago" if state['last_change_age'] is not None else "N/A"
                        print(f"  -> N/A: Never seen successful payload. Last attempt: {state['last_status']} ({state['last_error'] or 'no error'}) {age_str}")
                    else:
                        print("  -> N/A: Never seen successful payload (no attempts found)")
                else:
                    sa = f"{state['last_success_age']}s ago" if state['last_success_age'] is not None else "N/A"
                    ca = f"{state['last_change_age']}s ago" if state['last_change_age'] is not None else "N/A"
                    print(f"  Last Success: {sa} | Last Change: {ca} | Status: {state['last_status']}")
                    
                    keys = state["observed_keys"]
                    preview = keys[:10]
                    suffix = "..." if len(keys) > 10 else ""
                    print(f"  Keys ({len(keys)} observed): {preview} {suffix}")
                print()

            print("=" * 80)
            print("METRIC COMPUTABILITY")
            print("-" * 80)
            for mr in metric_results:
                status_str = f"[{mr['status']}]".ljust(12)
                print(f"{status_str} {mr['name']}")
                if mr['status'] == "COMPUTABLE":
                    print("  All required endpoints and keys are present.")
                else:
                    for reason in mr["reasons"]:
                        print(f"  Reason: {reason}")
                print()


def check_capabilities(catalog_path: str, db_path: str, plan_path: str, output_format: str = "text") -> None:
    checker = CapabilitiesChecker.from_paths(catalog_path, db_path, plan_path)
    checker.run(output_format)
