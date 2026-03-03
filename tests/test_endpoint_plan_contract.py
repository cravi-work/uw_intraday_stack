import copy

import pytest

from src.config_loader import load_endpoint_plan
from src.features import EXTRACTOR_REGISTRY
from src.ingest_engine import build_plan, summarize_effective_endpoint_plan


BASE_CFG = {
    "ingestion": {
        "watchlist": ["AAPL"],
        "cadence_minutes": 5,
        "enable_market_context": False,
    }
}


DISABLED_DEFAULT_NAMES = {
    "stock_volume_price_levels",
    "net_prem_ticks",
    "volume_oi_expiry",
    "interpolated_iv",
    "realized_vol",
    "option_stock_price_levels",
    "max_pain",
    "flow-alerts",
    "option-chains",
    "option-contracts",
}


DISABLED_DEFAULT_PATHS = {
    "/api/stock/{ticker}/stock-volume-price-levels",
    "/api/stock/{ticker}/net-prem-ticks",
    "/api/stock/{ticker}/option/volume-oi-expiry",
    "/api/stock/{ticker}/interpolated-iv",
    "/api/stock/{ticker}/volatility/realized",
    "/api/stock/{ticker}/option/stock-price-levels",
    "/api/stock/{ticker}/max-pain",
    "/api/stock/{ticker}/flow-alerts",
    "/api/stock/{ticker}/option-chains",
    "/api/stock/{ticker}/option-contracts",
}


MARKET_CONTEXT_PATHS = {
    "/api/market/market-tide",
    "/api/market/economic-calendar",
    "/api/market/top-net-impact",
    "/api/market/total-options-volume",
}



def test_endpoint_plan_requires_explicit_supported_purpose(tmp_path):
    missing = tmp_path / "missing.yaml"
    missing.write_text(
        "plans:\n  default:\n    - name: foo\n      method: GET\n      path: /api/foo\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing required 'purpose'"):
        load_endpoint_plan(missing)

    invalid = tmp_path / "invalid.yaml"
    invalid.write_text(
        "plans:\n  default:\n    - name: foo\n      purpose: maybe-later\n      method: GET\n      path: /api/foo\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="invalid purpose"):
        load_endpoint_plan(invalid)

    wrong_section = tmp_path / "wrong_section.yaml"
    wrong_section.write_text(
        "plans:\n  market_context:\n    - name: foo\n      purpose: signal-critical\n      method: GET\n      path: /api/market/foo\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="invalid for section 'market_context'"):
        load_endpoint_plan(wrong_section)



def test_default_plan_dry_run_excludes_disabled_overfetch_and_exposes_purposes():
    cfg = copy.deepcopy(BASE_CFG)
    plan = load_endpoint_plan("src/config/endpoint_plan.yaml")

    core, market = build_plan(cfg, plan)
    assert market == []

    core_paths = {call.path for call in core}
    core_purposes = {call.path: call.purpose for call in core}

    assert core_paths.isdisjoint(DISABLED_DEFAULT_PATHS)
    assert all(call.purpose in {"signal-critical", "report-only"} for call in core)
    assert core_paths.issubset(set(EXTRACTOR_REGISTRY.keys()))
    assert core_purposes["/api/darkpool/{ticker}"] == "report-only"
    assert len(core) == 17

    summary = summarize_effective_endpoint_plan(cfg, plan)
    assert {item["name"] for item in summary["disabled_default"]} == DISABLED_DEFAULT_NAMES
    assert {item["purpose"] for item in summary["fetched_default"]} == {"signal-critical", "report-only"}
    assert summary["fetched_market_context"] == []



def test_market_context_plan_fetches_only_context_endpoints_when_enabled():
    cfg = copy.deepcopy(BASE_CFG)
    cfg["ingestion"]["enable_market_context"] = True
    plan = load_endpoint_plan("src/config/endpoint_plan.yaml")

    _, market = build_plan(cfg, plan)
    market_paths = {call.path for call in market}

    assert market_paths == MARKET_CONTEXT_PATHS
    assert all(call.purpose == "context-only" for call in market)

    summary = summarize_effective_endpoint_plan(cfg, plan)
    assert {item["path"] for item in summary["fetched_market_context"]} == MARKET_CONTEXT_PATHS
    assert all(item["purpose"] == "context-only" for item in summary["fetched_market_context"])
