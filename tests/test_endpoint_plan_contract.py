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



DATE_REQUIRED_FETCHED_PATHS = {
    "/api/stock/{ticker}/ohlc/{candle_size}",
    "/api/stock/{ticker}/spot-exposures",
    "/api/stock/{ticker}/spot-exposures/strike",
    "/api/stock/{ticker}/greek-exposure",
    "/api/stock/{ticker}/greek-exposure/strike",
    "/api/stock/{ticker}/greek-exposure/expiry",
    "/api/stock/{ticker}/spot-exposures/expiry-strike",
    "/api/stock/{ticker}/flow-per-strike-intraday",
    "/api/stock/{ticker}/flow-per-strike",
    "/api/stock/{ticker}/oi-per-strike",
    "/api/stock/{ticker}/oi-change",
    "/api/stock/{ticker}/volatility/term-structure",
    "/api/stock/{ticker}/iv-rank",
    "/api/stock/{ticker}/historical-risk-reversal-skew",
    "/api/darkpool/{ticker}",
    "/api/lit-flow/{ticker}",
}

DATE_REQUIRED_MARKET_CONTEXT_PATHS = {
    "/api/market/market-tide",
    "/api/market/top-net-impact",
}

MARKET_CONTEXT_PATHS = {
    "/api/market/market-tide",
    "/api/market/economic-calendar",
    "/api/market/top-net-impact",
    "/api/market/total-options-volume",
}


def test_endpoint_plan_requires_explicit_supported_purpose_and_runtime_contract(tmp_path):
    missing = tmp_path / "missing.yaml"
    missing.write_text(
        "plans:\n  default:\n    - name: foo\n      method: GET\n      path: /api/foo\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing required 'purpose'"):
        load_endpoint_plan(missing)

    invalid = tmp_path / "invalid.yaml"
    invalid.write_text(
        "plans:\n  default:\n    - name: foo\n      purpose: maybe-later\n      decision_path: false\n      missing_affects_confidence: false\n      stale_affects_confidence: false\n      method: GET\n      path: /api/foo\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="invalid purpose"):
        load_endpoint_plan(invalid)

    wrong_section = tmp_path / "wrong_section.yaml"
    wrong_section.write_text(
        "plans:\n  market_context:\n    - name: foo\n      purpose: signal-critical\n      decision_path: true\n      missing_affects_confidence: true\n      stale_affects_confidence: true\n      method: GET\n      path: /api/market/foo\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="invalid for section 'market_context'"):
        load_endpoint_plan(wrong_section)

    missing_contract_field = tmp_path / "missing_contract_field.yaml"
    missing_contract_field.write_text(
        "plans:\n  default:\n    - name: foo\n      purpose: signal-critical\n      decision_path: true\n      missing_affects_confidence: true\n      method: GET\n      path: /api/foo\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing required 'stale_affects_confidence'"):
        load_endpoint_plan(missing_contract_field)

    incompatible_contract = tmp_path / "incompatible_contract.yaml"
    incompatible_contract.write_text(
        "plans:\n  default:\n    - name: foo\n      purpose: report-only\n      decision_path: true\n      missing_affects_confidence: false\n      stale_affects_confidence: false\n      method: GET\n      path: /api/foo\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="incompatible with purpose 'report-only'"):
        load_endpoint_plan(incompatible_contract)


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
    assert all(call.purpose_contract_version == "v1" for call in core)
    assert all(call.decision_path == (call.purpose == "signal-critical") for call in core)
    assert all(call.missing_affects_confidence == call.decision_path for call in core)
    assert all(call.stale_affects_confidence == call.decision_path for call in core)
    assert len(core) == 17
    assert {
        call.path for call in core
        if call.query_params.get("date") == "{date}"
    }.issuperset(DATE_REQUIRED_FETCHED_PATHS)

    summary = summarize_effective_endpoint_plan(cfg, plan)
    assert {item["name"] for item in summary["disabled_default"]} == DISABLED_DEFAULT_NAMES
    assert {item["purpose"] for item in summary["fetched_default"]} == {"signal-critical", "report-only"}
    assert all(item["purpose_contract_version"] == "v1" for item in summary["fetched_default"])
    assert all(item["decision_path"] == (item["purpose"] == "signal-critical") for item in summary["fetched_default"])
    assert all(item["missing_affects_confidence"] == item["decision_path"] for item in summary["fetched_default"])
    assert all(item["stale_affects_confidence"] == item["decision_path"] for item in summary["fetched_default"])
    assert summary["fetched_market_context"] == []


def test_market_context_plan_fetches_only_context_endpoints_when_enabled():
    cfg = copy.deepcopy(BASE_CFG)
    cfg["ingestion"]["enable_market_context"] = True
    plan = load_endpoint_plan("src/config/endpoint_plan.yaml")

    _, market = build_plan(cfg, plan)
    market_paths = {call.path for call in market}

    assert market_paths == MARKET_CONTEXT_PATHS
    assert all(call.purpose == "context-only" for call in market)
    assert all(call.decision_path is False for call in market)
    assert all(call.missing_affects_confidence is False for call in market)
    assert all(call.stale_affects_confidence is False for call in market)
    assert {
        call.path for call in market
        if call.query_params.get("date") == "{date}"
    }.issuperset(DATE_REQUIRED_MARKET_CONTEXT_PATHS)

    summary = summarize_effective_endpoint_plan(cfg, plan)
    assert {item["path"] for item in summary["fetched_market_context"]} == MARKET_CONTEXT_PATHS
    assert all(item["purpose"] == "context-only" for item in summary["fetched_market_context"])
    assert all(item["decision_path"] is False for item in summary["fetched_market_context"])
    assert all(item["missing_affects_confidence"] is False for item in summary["fetched_market_context"])
    assert all(item["stale_affects_confidence"] is False for item in summary["fetched_market_context"])
