import json
from datetime import datetime, timezone
from typing import Any, Dict, Set

import duckdb
import pytest

from src.capabilities import flatten_keys, CapabilitiesChecker, EndpointTruth, CapabilitiesError
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
    assert "a.c" in out1
    assert "a.d" in out1
    assert "b" in out1
    assert "list.[].x" in out1
    assert "list.[].y" in out1
    assert "a" not in out1
    assert "list" not in out1


def setup_in_memory_db() -> duckdb.DuckDBPyConnection:
    # [Fix: Step 9] Strict column matching to production schema
    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE dim_endpoints (endpoint_id INTEGER, method TEXT, path TEXT)")
    con.execute("CREATE TABLE endpoint_state (ticker TEXT, endpoint_id INTEGER, last_success_event_id TEXT, last_success_ts_utc TIMESTAMP, last_change_ts_utc TIMESTAMP)")
    con.execute("CREATE TABLE raw_http_events (event_id TEXT, endpoint_id INTEGER, received_at_utc TIMESTAMP, http_status INTEGER, error_type TEXT, payload_json JSON)")
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
    con.execute("INSERT INTO endpoint_state VALUES ('AAPL', 1, 'evt-1', '2025-01-01 10:00:00', '2025-01-01 10:00:00')")
    con.execute("INSERT INTO raw_http_events VALUES ('evt-1', 1, '2025-01-01 10:00:00', 200, NULL, '{\"status\": \"ok\"}')")

    checker = CapabilitiesChecker(
        catalog=DummyCatalog(),
        plan_yaml={"plans": {"default": []}},
        db_path=":memory:",
        validate_db_schema=False
    )
    
    truth = checker.extract_db_truth(con, "GET", "/api/test")
    assert truth.has_success is True
    assert truth.last_status == 200
    assert "status" in truth.observed_keys
    assert truth.selected_endpoint_id == 1
    assert truth.selected_success_event_id == "evt-1"


def test_determinism_multi_ticker():
    """[Fix: Step 9] Asserts multi-ticker endpoint deterministically returns the latest success globally."""
    con = setup_in_memory_db()
    con.execute("INSERT INTO dim_endpoints VALUES (1, 'GET', '/api/multi')")
    
    # Older success (AAPL)
    con.execute("INSERT INTO endpoint_state VALUES ('AAPL', 1, 'evt-aapl', '2025-01-01 09:00:00', '2025-01-01 09:00:00')")
    con.execute("INSERT INTO raw_http_events VALUES ('evt-aapl', 1, '2025-01-01 09:00:00', 200, NULL, '{\"old\": 1}')")
    
    # Newer success (MSFT)
    con.execute("INSERT INTO endpoint_state VALUES ('MSFT', 1, 'evt-msft', '2025-01-01 10:00:00', '2025-01-01 10:00:00')")
    con.execute("INSERT INTO raw_http_events VALUES ('evt-msft', 1, '2025-01-01 10:00:00', 200, NULL, '{\"new\": 2}')")

    checker = CapabilitiesChecker(catalog=DummyCatalog(), plan_yaml={"plans": {}}, db_path=":memory:", validate_db_schema=False)
    truth = checker.extract_db_truth(con, "GET", "/api/multi")
    
    assert "new" in truth.observed_keys
    assert "old" not in truth.observed_keys
    assert truth.selected_success_event_id == "evt-msft"


def test_variant_ordering():
    """[Fix: Step 9] Asserts variant resolution strictly orders by newest global success event across variants."""
    con = setup_in_memory_db()
    
    # Variant 1: Oldest endpoint definition, newest success
    con.execute("INSERT INTO dim_endpoints VALUES (1, 'GET', '/api/variant')")
    con.execute("INSERT INTO endpoint_state VALUES ('AAPL', 1, 'evt-v1', '2025-01-01 10:00:00', '2025-01-01 10:00:00')")
    con.execute("INSERT INTO raw_http_events VALUES ('evt-v1', 1, '2025-01-01 10:00:00', 200, NULL, '{\"v1\": 1}')")
    
    # Variant 2: Newer endpoint definition, older success
    con.execute("INSERT INTO dim_endpoints VALUES (2, 'GET', '/api/variant')")
    con.execute("INSERT INTO endpoint_state VALUES ('AAPL', 2, 'evt-v2', '2025-01-01 09:00:00', '2025-01-01 09:00:00')")
    con.execute("INSERT INTO raw_http_events VALUES ('evt-v2', 2, '2025-01-01 09:00:00', 200, NULL, '{\"v2\": 1}')")

    checker = CapabilitiesChecker(catalog=DummyCatalog(), plan_yaml={"plans": {}}, db_path=":memory:", validate_db_schema=False)
    truth = checker.extract_db_truth(con, "GET", "/api/variant")
    
    assert truth.selected_endpoint_id == 1
    assert "v1" in truth.observed_keys
    assert "v2" not in truth.observed_keys


def test_failure_visibility():
    """Verify that if no success exists, we still see the last failure status"""
    con = setup_in_memory_db()
    con.execute("INSERT INTO dim_endpoints VALUES (1, 'GET', '/api/fail')")
    con.execute("INSERT INTO endpoint_state VALUES ('AAPL', 1, NULL, NULL, '2025-01-01 12:00:00')")
    con.execute("INSERT INTO raw_http_events VALUES ('evt-fail', 1, '2025-01-01 12:00:00', 500, 'InternalError', NULL)")

    checker = CapabilitiesChecker(
        catalog=DummyCatalog(),
        plan_yaml={"plans": {"default": []}},
        db_path=":memory:",
        validate_db_schema=False
    )
    
    truth = checker.extract_db_truth(con, "GET", "/api/fail")
    assert truth.has_success is False
    assert truth.last_status == 500
    assert truth.last_error == "InternalError"
    assert truth.last_success_age_s is None
    assert truth.last_change_age_s is not None


def test_metric_computability_strictness():
    """[Fix: Step 7] Metric fails if required endpoint has no key rules and is NOT in presence_only_endpoints"""
    checker = CapabilitiesChecker(
        catalog=DummyCatalog(),
        plan_yaml={"plans": {"default": []}},
        db_path=":memory:",
        validate_db_schema=False
    )

    ep = EndpointRef("GET", "/api/data")
    spec = MetricSpec(
        name="StrictMetric",
        required_endpoints=[ep],
        required_keys_by_endpoint={},
        presence_only_endpoints=set()
    )

    db_states = {ep: EndpointTruth(True, 200, None, 10, 10, ["any"], 1, "evt-1")}
    
    res = checker.evaluate_metric(spec, db_states)
    assert res["status"] == "N/A"
    assert any("no key rules defined" in r for r in res["reasons"])


def test_metric_presence_only_logic():
    """[Fix: Step 7] Metric passes with COMPUTABLE_PRESENCE_ONLY flag if fallback is explicit"""
    checker = CapabilitiesChecker(
        catalog=DummyCatalog(),
        plan_yaml={"plans": {"default": []}},
        db_path=":memory:",
        validate_db_schema=False
    )

    ep = EndpointRef("GET", "/api/data")
    spec = MetricSpec(
        name="PresenceMetric",
        required_endpoints=[ep],
        required_keys_by_endpoint={},
        presence_only_endpoints={ep}
    )

    db_states = {ep: EndpointTruth(True, 200, None, 10, 10, ["any"], 1, "evt-1")}
    
    res = checker.evaluate_metric(spec, db_states)
    assert res["status"] == "COMPUTABLE_PRESENCE_ONLY"


def test_metric_computability_logic():
    """Metric passes/fails correctly based on required keys"""
    checker = CapabilitiesChecker(
        catalog=DummyCatalog(),
        plan_yaml={"plans": {"default": []}},
        db_path=":memory:",
        validate_db_schema=False
    )

    ep = EndpointRef("GET", "/api/data")
    spec = MetricSpec(
        name="TestMetric",
        required_endpoints=[ep],
        required_keys_by_endpoint={ep: KeyRule(all_of={"req_1"})},
        presence_only_endpoints=set()
    )

    db_states_fail = {ep: EndpointTruth(True, 200, None, 10, 10, ["wrong_key"], 1, "evt-1")}
    res_fail = checker.evaluate_metric(spec, db_states_fail)
    assert res_fail["status"] == "N/A"
    assert any("missing required keys" in r for r in res_fail["reasons"])

    db_states_pass = {ep: EndpointTruth(True, 200, None, 10, 10, ["req_1", "extra"], 1, "evt-1")}
    res_pass = checker.evaluate_metric(spec, db_states_pass)
    assert res_pass["status"] == "COMPUTABLE"
    assert not res_pass["reasons"]


def test_in_plan_tiers_reporting():
    """[Fix: Step 8] Verify tier detection mapping logic works accurately"""
    plan = {
        "plans": {
            "default": [{"method": "POST", "path": "/api/in_plan"}],
            "market_context": [{"method": "POST", "path": "/api/in_plan"}]
        }
    }
    checker = CapabilitiesChecker(catalog=DummyCatalog(), plan_yaml=plan, db_path=":memory:", validate_db_schema=False)
    
    key = ("POST", "/api/in_plan")
    assert key in checker.planned_endpoints
    assert set(checker.planned_endpoints[key]) == {"default", "market_context"}
    assert ("GET", "/api/missing") not in checker.planned_endpoints