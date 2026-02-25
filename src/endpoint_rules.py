from __future__ import annotations
import logging
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, Any

logger = logging.getLogger(__name__)

class EmptyPayloadPolicy(Enum):
    EMPTY_IS_DATA = "EMPTY_IS_DATA"
    EMPTY_MEANS_STALE = "EMPTY_MEANS_STALE"
    EMPTY_INVALID = "EMPTY_INVALID"

@dataclass(frozen=True)
class EndpointRule:
    method: str
    path: str
    empty_policy_by_session: Dict[str, EmptyPayloadPolicy]
    required_any_keys: Optional[Tuple[str, ...]] = None
    required_all_keys: Optional[Tuple[str, ...]] = None
    data_container_keys: Tuple[str, ...] = ("data", "results", "items", "trades", "history")

RULE_REGISTRY: Dict[Tuple[str, str], EndpointRule] = {}

def _register(rule: EndpointRule) -> None:
    RULE_REGISTRY[(rule.method.upper(), rule.path)] = rule

P_ALWAYS_DATA = {
    "PREMARKET": EmptyPayloadPolicy.EMPTY_IS_DATA, 
    "RTH": EmptyPayloadPolicy.EMPTY_IS_DATA, 
    "AFTERHOURS": EmptyPayloadPolicy.EMPTY_IS_DATA, 
    "CLOSED": EmptyPayloadPolicy.EMPTY_IS_DATA
}

P_FLOW = {
    "PREMARKET": EmptyPayloadPolicy.EMPTY_MEANS_STALE, 
    "RTH": EmptyPayloadPolicy.EMPTY_IS_DATA, 
    "AFTERHOURS": EmptyPayloadPolicy.EMPTY_MEANS_STALE, 
    "CLOSED": EmptyPayloadPolicy.EMPTY_MEANS_STALE
}

P_STRUCTURAL = {
    "PREMARKET": EmptyPayloadPolicy.EMPTY_MEANS_STALE, 
    "RTH": EmptyPayloadPolicy.EMPTY_INVALID, 
    "AFTERHOURS": EmptyPayloadPolicy.EMPTY_MEANS_STALE, 
    "CLOSED": EmptyPayloadPolicy.EMPTY_MEANS_STALE
}

for p in ["/api/stock/{ticker}/flow-alerts", "/api/darkpool/{ticker}", "/api/lit-flow/{ticker}"]:
    _register(EndpointRule("GET", p, P_ALWAYS_DATA))

for p in ["/api/stock/{ticker}/flow-per-strike", "/api/stock/{ticker}/flow-per-strike-intraday", "/api/stock/{ticker}/flow-recent", "/api/stock/{ticker}/net-prem-ticks"]:
    _register(EndpointRule("GET", p, P_FLOW))

for p in [
    "/api/stock/{ticker}/oi-per-strike", "/api/stock/{ticker}/oi-change", "/api/stock/{ticker}/option/volume-oi-expiry",
    "/api/stock/{ticker}/option-chains", "/api/stock/{ticker}/option-contracts", "/api/stock/{ticker}/volatility/term-structure",
    "/api/stock/{ticker}/interpolated-iv", "/api/stock/{ticker}/iv-rank", "/api/stock/{ticker}/historical-risk-reversal-skew",
    "/api/stock/{ticker}/greek-exposure", "/api/stock/{ticker}/greek-exposure/strike", "/api/stock/{ticker}/greek-exposure/expiry",
    "/api/stock/{ticker}/spot-exposures", "/api/stock/{ticker}/spot-exposures/strike", "/api/stock/{ticker}/spot-exposures/expiry-strike",
    "/api/stock/{ticker}/max-pain",
    "/api/market/market-tide", "/api/market/economic-calendar", "/api/market/top-net-impact", "/api/market/total-options-volume"
]:
    _register(EndpointRule("GET", p, P_STRUCTURAL))

def get_endpoint_rule(method: str, path: str) -> Optional[EndpointRule]:
    return RULE_REGISTRY.get((method.upper(), path))

def get_empty_policy(method: str, path: str, session_label: str) -> EmptyPayloadPolicy:
    rule = get_endpoint_rule(method, path)
    if not rule:
        return EmptyPayloadPolicy.EMPTY_INVALID
        
    legacy_map = {
        "PRE": "PREMARKET",
        "REG": "RTH",
        "AFT": "AFTERHOURS"
    }
    
    canonical_label = legacy_map.get(session_label, session_label)
    
    if canonical_label not in ["PREMARKET", "RTH", "AFTERHOURS", "CLOSED"]:
        logger.error(f"Session contract violation: Unknown session label '{session_label}' provided to get_empty_policy.")
        raise ValueError(f"Unknown session label: {session_label}")
        
    return rule.empty_policy_by_session.get(canonical_label, EmptyPayloadPolicy.EMPTY_INVALID)

def validate_plan_coverage(plan_yaml: Dict[str, Any]) -> None:
    missing = []
    for tier, eps in plan_yaml.get("plans", {}).items():
        for ep in eps:
            method = ep.get("method", "GET").upper()
            path = ep.get("path", "")
            if (method, path) not in RULE_REGISTRY:
                missing.append(f"{method} {path}")
    
    if missing:
        raise RuntimeError(f"CRITICAL: Missing explicit endpoint rules for: {missing}. Define them in src/endpoint_rules.py to prevent silent defaults.")