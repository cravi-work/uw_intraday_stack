from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional


@dataclass(frozen=True)
class EndpointRef:
    method: str
    path: str


@dataclass(frozen=True)
class KeyRule:
    all_of: Set[str] = field(default_factory=set)
    any_of: List[Set[str]] = field(default_factory=list)


@dataclass(frozen=True)
class MetricSpec:
    name: str
    required_endpoints: List[EndpointRef]
    required_keys_by_endpoint: Dict[EndpointRef, KeyRule]
    # [Fix: Step 7] Explicit presence-only endpoints to prevent silent passes
    presence_only_endpoints: Set[EndpointRef] = field(default_factory=set)


# Institutional seed set: explicitly curated, minimal dependencies based on actual extractors
INSTITUTIONAL_METRICS: List[MetricSpec] = [
    MetricSpec(
        name="Gamma/Dealer",
        required_endpoints=[
            EndpointRef("GET", "/api/stock/{ticker}/greek-exposure/strike"),
            EndpointRef("GET", "/api/stock/{ticker}/greek-exposure/expiry"),
            EndpointRef("GET", "/api/stock/{ticker}/spot-exposures/expiry-strike"),
        ],
        required_keys_by_endpoint={
            EndpointRef("GET", "/api/stock/{ticker}/greek-exposure/strike"): KeyRule(
                any_of=[{"data.[].strike", "data.[].gamma_exposure"}, {"data.[].strike", "data.[].gex"}]
            )
        },
    ),
    MetricSpec(
        name="Flow",
        required_endpoints=[
            EndpointRef("GET", "/api/stock/{ticker}/flow-per-strike"),
            EndpointRef("GET", "/api/stock/{ticker}/flow-alerts"),
        ],
        required_keys_by_endpoint={
            EndpointRef("GET", "/api/stock/{ticker}/flow-alerts"): KeyRule(
                all_of={"data.[].premium", "data.[].dte", "data.[].side", "data.[].put_call"}
            )
        },
    ),
    MetricSpec(
        name="Options Surface",
        required_endpoints=[
            EndpointRef("GET", "/api/stock/{ticker}/option-chains"),
            EndpointRef("GET", "/api/stock/{ticker}/option-contracts"),
            EndpointRef("GET", "/api/stock/{ticker}/historical-risk-reversal-skew"),
        ],
        required_keys_by_endpoint={},
        # [Fix: Step 7] Explicitly denote volatile endpoints as presence-only 
        presence_only_endpoints={
            EndpointRef("GET", "/api/stock/{ticker}/option-chains"),
            EndpointRef("GET", "/api/stock/{ticker}/option-contracts"),
            EndpointRef("GET", "/api/stock/{ticker}/historical-risk-reversal-skew"),
        },
    ),
    MetricSpec(
        name="Market Context",
        required_endpoints=[
            EndpointRef("GET", "/api/market/top-net-impact"),
            EndpointRef("GET", "/api/market/total-options-volume"),
        ],
        required_keys_by_endpoint={},
        # [Fix: Step 7]
        presence_only_endpoints={
            EndpointRef("GET", "/api/market/top-net-impact"),
            EndpointRef("GET", "/api/market/total-options-volume"),
        },
    ),
]