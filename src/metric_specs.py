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
    # Explicit presence-only endpoints to prevent silent passes
    presence_only_endpoints: Set[EndpointRef] = field(default_factory=set)


# Institutional seed set: strictly matched against real-world payload schemas
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
                # Corrected to match true API keys
                any_of=[{"data.[].strike", "data.[].call_gex", "data.[].put_gex"}]
            )
        },
        presence_only_endpoints={
            EndpointRef("GET", "/api/stock/{ticker}/greek-exposure/expiry"),
            EndpointRef("GET", "/api/stock/{ticker}/spot-exposures/expiry-strike")
        }
    ),
    MetricSpec(
        name="Flow",
        required_endpoints=[
            EndpointRef("GET", "/api/stock/{ticker}/flow-per-strike"),
            EndpointRef("GET", "/api/stock/{ticker}/flow-alerts"),
        ],
        required_keys_by_endpoint={},
        # Marked as presence-only until complex nested array parsing is verified
        presence_only_endpoints={
            EndpointRef("GET", "/api/stock/{ticker}/flow-per-strike"),
            EndpointRef("GET", "/api/stock/{ticker}/flow-alerts")
        }
    ),
    MetricSpec(
        name="Options Surface",
        required_endpoints=[
            EndpointRef("GET", "/api/stock/{ticker}/option-chains"),
            EndpointRef("GET", "/api/stock/{ticker}/option-contracts"),
            EndpointRef("GET", "/api/stock/{ticker}/historical-risk-reversal-skew"),
        ],
        required_keys_by_endpoint={},
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
        presence_only_endpoints={
            EndpointRef("GET", "/api/market/top-net-impact"),
            EndpointRef("GET", "/api/market/total-options-volume"),
        },
    ),
]