from __future__ import annotations
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

class EmptyPayloadPolicy(Enum):
    """
    EMPTY_IS_DATA: empty payload is a valid measurement ("zero events") and should update last_success_*
    EMPTY_MEANS_STALE: empty payload is not data (session/venue limitation); should not overwrite last success; triggers stale carry-forward
    EMPTY_INVALID: empty payload indicates provider or query error; treat as error
    """
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

# --- Policy Definitions ---
P_ALWAYS_DATA = {
    "PRE": EmptyPayloadPolicy.EMPTY_IS_DATA, 
    "REG": EmptyPayloadPolicy.EMPTY_IS_DATA, 
    "AFT": EmptyPayloadPolicy.EMPTY_IS_DATA, 
    "CLOSED": EmptyPayloadPolicy.EMPTY_IS_DATA
}

P_FLOW = {
    "PRE": EmptyPayloadPolicy.EMPTY_MEANS_STALE, 
    "REG": EmptyPayloadPolicy.EMPTY_IS_DATA, 
    "AFT": EmptyPayloadPolicy.EMPTY_MEANS_STALE, 
    "CLOSED": EmptyPayloadPolicy.EMPTY_MEANS_STALE
}

P_STRUCTURAL = {
    "PRE": EmptyPayloadPolicy.EMPTY_MEANS_STALE, 
    "REG": EmptyPayloadPolicy.EMPTY_INVALID, 
    "AFT": EmptyPayloadPolicy.EMPTY_MEANS_STALE, 
    "CLOSED": EmptyPayloadPolicy.EMPTY_MEANS_STALE
}

# 1. ALWAYS DATA
for p in [
    "/api/stock/{ticker}/flow-alerts", 
    "/api/darkpool/{ticker}", 
    "/api/lit-flow/{ticker}"
]:
    _register(EndpointRule("GET", p, P_ALWAYS_DATA))

# 2. FLOW (Session-Aware Empties)
for p in [
    "/api/stock/{ticker}/flow-per-strike", 
    "/api/stock/{ticker}/flow-per-strike-intraday", 
    "/api/stock/{ticker}/flow-recent", 
    "/api/stock/{ticker}/net-prem-ticks"
]:
    _register(EndpointRule("GET", p, P_FLOW))

# 3. STRUCTURAL OPTIONS (Never empty in REG, Stale in PRE/AFT)
for p in [
    "/api/stock/{ticker}/oi-per-strike",
    "/api/stock/{ticker}/oi-change",
    "/api/stock/{ticker}/option/volume-oi-expiry",
    "/api/stock/{ticker}/option-chains",
    "/api/stock/{ticker}/option-contracts",
    "/api/stock/{ticker}/volatility/term-structure",
    "/api/stock/{ticker}/interpolated-iv",
    "/api/stock/{ticker}/iv-rank",
    "/api/stock/{ticker}/historical-risk-reversal-skew",
    "/api/stock/{ticker}/greek-exposure",
    "/api/stock/{ticker}/greek-exposure/strike",
    "/api/stock/{ticker}/greek-exposure/expiry",
    "/api/stock/{ticker}/spot-exposures",
    "/api/stock/{ticker}/spot-exposures/strike",
    "/api/stock/{ticker}/spot-exposures/expiry-strike",
    "/api/stock/{ticker}/max-pain",
    "/api/market/sectors",
    "/api/market/indices",
    "/api/market/market-context"
]:
    _register(EndpointRule("GET", p, P_STRUCTURAL))


def get_endpoint_rule(method: str, path: str) -> Optional[EndpointRule]:
    return RULE_REGISTRY.get((method.upper(), path))

def get_empty_policy(method: str, path: str, session_label: str) -> EmptyPayloadPolicy:
    rule = get_endpoint_rule(method, path)
    if not rule:
        return EmptyPayloadPolicy.EMPTY_INVALID
    return rule.empty_policy_by_session.get(session_label, EmptyPayloadPolicy.EMPTY_INVALID)