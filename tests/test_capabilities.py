import json
from datetime import datetime, timezone
from typing import Any, Dict, Set

import duckdb
import pytest

from src.capabilities import flatten_keys, CapabilitiesChecker
from src.metric_specs import EndpointRef, KeyRule, MetricSpec

UTC = timezone.utc

def test_flatten_keys_determinism():
    """same JSON different insertion order -> same output, leaf nodes only"""
    dict1 = {"b": 2, "a": {"d": 4, "c": 3}, "list": [{"y": 2, "x": 1}]}
    dict2 = {"list": [{"x": 1, "y": 2}], "a": {"c": 3, "d": 4}, "b": 2}
    
    out1: Set[str] = set()
    flatten_keys(dict1, "", 5, out1)
    
    out2: Set[str] = set()
    flatten_keys(dict2, "", 5, out2)
    
    assert out1 == out2
    # Assert only leaf paths exist (no "a" or "list")
    assert "a.c" in out1
    assert "a.d" in out1
    assert "b" in out1
    assert "list.[].x" in out1
    assert "list.[].y" in out1
    assert "a" not in out1
    assert "list" not in out1

def setup_in_memory_db() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE dim_endpoints (endpoint_id INTEGER, method TEXT, path TEXT)")
    con.execute("CREATE TABLE endpoint_state (endpoint_id INTEGER, last_success_event_id TEXT, last_success_ts_utc TIMESTAMP, last_change_ts_utc TIMESTAMP)")
    con.execute("CREATE TABLE raw_http_events (event_id TEXT, endpoint_id INTEGER, http_status INTEGER, error_type TEXT, payload_json JSON, received_at_utc TIMESTAMP)")
    return con

class DummyCatalog:
    def has(self, m, p): return True
    def get(self, m, p): return type("obj", (object,), {"operation_id": "op1"})()
    @property
    def catalog_hash(self): return "hash"

def test_endpoint_status_resolution():
    """Verify endpoint_state logic correctly follows the canonical path"""
    con = setup_in_memory_db()
    con.execute("INSERT INTO dim_endpoints VALUES (1, 'GET', '/api/test')")
    con.execute("INSERT INTO endpoint_state VALUES (1, 'evt-1', '2025-01-01 10:00:00', '2025-01-01 10:00:00')")
    con.execute("INSERT INTO raw_http_events VALUES ('evt-1', 1, 200, NULL, '{\"status\": \"ok\"}', '2025-01-01 10:00:00')")

    checker = CapabilitiesChecker(
        catalog=DummyCatalog(),
        plan_yaml={"plans": {"default": []}},
        db_path=":memory:"
    )
    
    truth = checker.extract_db_truth(con, "GET", "/api/test")
    assert truth["has_success"] is True
    assert truth["last_status"] == 200
    assert "status" in truth["observed_keys"]

def test_failure_visibility():
    """Verify that if no success exists, we still see the last failure status"""
    con = setup_in_memory_db()
    con.execute("INSERT INTO dim_endpoints VALUES (1, 'GET', '/api/fail')")
    # No success in endpoint_state
    con.execute("INSERT INTO endpoint_state VALUES (1, NULL, NULL, '2025-01-01 12:00:00')")
    # Raw event is a 500
    con.execute("INSERT INTO raw_http_events VALUES ('evt-fail', 1, 500, 'InternalError', NULL, '2025-01-01 12:00:00')")

    checker = CapabilitiesChecker(
        catalog=DummyCatalog(),
        plan_yaml={"plans": {"default": []}},
        db_path=":memory:"
    )
    
    truth = checker.extract_db_truth(con, "GET", "/api/fail")
    assert truth["has_success"] is False
    assert truth["last_status"] == 500
    assert truth["last_error"] == "InternalError"
    assert truth["last_success_age"] is None
    assert truth["last_change_age"] is not None

def test_metric_computability_strictness():
    """Metric fails if required endpoint has no key rules (strict default)"""
    checker = CapabilitiesChecker(
        catalog=DummyCatalog(),
        plan_yaml={"plans": {"default": []}},
        db_path=":memory:"
    )

    ep = EndpointRef("GET", "/api/data")
    # Spec with NO rules for the endpoint
    spec = MetricSpec(
        name="StrictMetric",
        required_endpoints=[ep],
        required_keys_by_endpoint={} 
    )

    # DB has success
    db_states = {ep: {"has_success": True, "observed_keys": ["any"]}}
    
    res = checker.evaluate_metric(spec, db_states)
    assert res["status"] == "N/A"
    assert any("no key rules defined" in r for r in res["reasons"])

def test_metric_computability_logic():
    """Metric passes/fails correctly based on required keys"""
    checker = CapabilitiesChecker(
        catalog=DummyCatalog(),
        plan_yaml={"plans": {"default": []}},
        db_path=":memory:"
    )

    ep = EndpointRef("GET", "/api/data")
    spec = MetricSpec(
        name="TestMetric",
        required_endpoints=[ep],
        required_keys_by_endpoint={ep: KeyRule(all_of={"req_1"})}
    )

    # Fail scenario
    db_states_fail = {ep: {"has_success": True, "observed_keys": ["wrong_key"]}}
    res_fail = checker.evaluate_metric(spec, db_states_fail)
    assert res_fail["status"] == "N/A"
    assert any("missing required keys" in r for r in res_fail["reasons"])

    # Pass scenario
    db_states_pass = {ep: {"has_success": True, "observed_keys": ["req_1", "extra"]}}
    res_pass = checker.evaluate_metric(spec, db_states_pass)
    assert res_pass["status"] == "COMPUTABLE"
    assert not res_pass["reasons"]
