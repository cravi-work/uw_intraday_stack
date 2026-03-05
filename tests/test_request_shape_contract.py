import copy
import logging

from src.api_catalog_loader import load_api_catalog
from src.config_loader import load_endpoint_plan
from src.endpoint_rules import validate_plan_coverage
from src.ingest_engine import build_plan


BASE_CFG = {
    "ingestion": {
        "watchlist": ["AAPL"],
        "cadence_minutes": 5,
        "enable_market_context": True,
    }
}


def test_build_plan_injects_date_for_known_date_capable_snapshot_endpoints():
    cfg = copy.deepcopy(BASE_CFG)
    plan = {
        "plans": {
            "default": [
                {
                    "name": "ohlc_1m",
                    "purpose": "signal-critical",
                    "decision_path": True,
                    "missing_affects_confidence": True,
                    "stale_affects_confidence": True,
                    "method": "GET",
                    "path": "/api/stock/{ticker}/ohlc/{candle_size}",
                    "path_params": {"candle_size": "1m"},
                },
                {
                    "name": "greek-exposure-strike",
                    "purpose": "signal-critical",
                    "decision_path": True,
                    "missing_affects_confidence": True,
                    "stale_affects_confidence": True,
                    "method": "GET",
                    "path": "/api/stock/{ticker}/greek-exposure/strike",
                },
                {
                    "name": "flow-per-strike",
                    "purpose": "signal-critical",
                    "decision_path": True,
                    "missing_affects_confidence": True,
                    "stale_affects_confidence": True,
                    "method": "GET",
                    "path": "/api/stock/{ticker}/flow-per-strike",
                },
                {
                    "name": "historical-risk-reversal-skew",
                    "purpose": "signal-critical",
                    "decision_path": True,
                    "missing_affects_confidence": True,
                    "stale_affects_confidence": True,
                    "method": "GET",
                    "path": "/api/stock/{ticker}/historical-risk-reversal-skew",
                },
            ],
            "market_context": [
                {
                    "name": "market-top-net-impact",
                    "purpose": "context-only",
                    "decision_path": False,
                    "missing_affects_confidence": False,
                    "stale_affects_confidence": False,
                    "method": "GET",
                    "path": "/api/market/top-net-impact",
                }
            ],
        }
    }

    core, market = build_plan(cfg, plan)
    assert all(call.query_params.get("date") == "{date}" for call in core)
    assert all(call.query_params.get("date") == "{date}" for call in market)



def test_current_plan_request_shape_matches_catalog_and_has_explicit_ohlc_rule(caplog):
    plan = load_endpoint_plan("src/config/endpoint_plan.yaml")
    registry = load_api_catalog("api_catalog.generated.yaml")

    with caplog.at_level(logging.WARNING):
        validate_plan_coverage(plan)
    assert "/api/stock/{ticker}/ohlc/{candle_size}" not in caplog.text

    cfg = copy.deepcopy(BASE_CFG)
    core, market = build_plan(cfg, plan)
    date_required_paths = {
        "/api/stock/{ticker}/ohlc/{candle_size}",
        "/api/stock/{ticker}/greek-exposure/strike",
        "/api/stock/{ticker}/greek-exposure/expiry",
        "/api/stock/{ticker}/spot-exposures/expiry-strike",
        "/api/stock/{ticker}/flow-per-strike",
        "/api/stock/{ticker}/historical-risk-reversal-skew",
        "/api/market/top-net-impact",
    }

    for call in core + market:
        if call.path in date_required_paths:
            assert call.query_params.get("date") == "{date}"
            assert "date" in registry.allowed_query_params(call.method, call.path)


def test_market_tide_bool_query_flags_are_coerced_to_ints():
    plan = load_endpoint_plan("src/config/endpoint_plan.yaml")
    cfg = copy.deepcopy(BASE_CFG)
    _core, market = build_plan(cfg, plan)

    market_tide = next((c for c in market if c.path == "/api/market/market-tide"), None)
    assert market_tide is not None, "market-tide must be present when market context is enabled"
    # endpoint_rules.normalize_runtime_query_params coerces bools to 0/1 to avoid 400s for
    # endpoints that model boolean-like flags numerically.
    assert market_tide.query_params.get("interval_5m") == 1
