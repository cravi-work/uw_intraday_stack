from src.config_loader import load_endpoint_plan, load_yaml
from src.ingest_engine import build_plan, _expand_path_params_for_call

def test_ticker_injected_for_paths():
    cfg = load_yaml("src/config/config.yaml").raw
    plan_yaml = load_endpoint_plan("src/config/endpoint_plan.yaml")
    core, _ = build_plan(cfg, plan_yaml)
    call = [c for c in core if "{ticker}" in c.path][0]
    pp = _expand_path_params_for_call(call, ticker="SPY", date="2026-02-16")
    assert pp["ticker"] == "SPY"
