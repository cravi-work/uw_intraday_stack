from __future__ import annotations

import sys
import types
from pathlib import Path


def test_endpoint_plan_no_interval_5m_for_market_tide() -> None:
    plan_path = Path(__file__).resolve().parents[1] / "src" / "config" / "endpoint_plan.yaml"
    text = plan_path.read_text()
    assert "interval_5m" not in text


def test_market_tide_retry_sequence_drops_interval_then_date() -> None:
    # ingest_engine imports duckdb at module import time; stub it for unit tests.
    sys.modules.setdefault("duckdb", types.ModuleType("duckdb"))

    from src.ingest_engine import _market_tide_retry_query_params

    qp = {"date": "2026-03-06", "interval_5m": True}
    seq = _market_tide_retry_query_params(qp)

    assert seq == [{"date": "2026-03-06"}, None]
