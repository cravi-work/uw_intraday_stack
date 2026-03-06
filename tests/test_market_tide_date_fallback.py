import sys
import time
import types

import pytest


# The project uses DuckDB for persistence, but unit tests that exercise pure ingest
# logic should still run even if DuckDB is not present in the environment.
try:  # pragma: no cover
    import duckdb  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    sys.modules["duckdb"] = types.ModuleType("duckdb")

try:  # pragma: no cover
    import aiolimiter  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    m = types.ModuleType("aiolimiter")

    class AsyncLimiter:  # minimal shim for import-time use
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    m.AsyncLimiter = AsyncLimiter
    sys.modules["aiolimiter"] = m


from src.ingest_engine import PlannedCall, fetch_all
from src.uw_client import HttpResult


def _hr(*, status_code: int, ok: bool, payload_json=None, payload_text=None) -> HttpResult:
    now = time.time()
    return HttpResult(
        ok=ok,
        status_code=status_code,
        payload_json=payload_json,
        payload_text=payload_text,
        payload_hash=None,
        requested_at_utc=now,
        received_at_utc=now,
        latency_ms=1,
        error_type=None if ok else "http_error",
        error_message=None if ok else "unprocessable",
        retry_count=0,
        response_headers=None,
    )


class _StubClient:
    def __init__(self):
        self.calls = []
        self._n = 0

    async def request(self, method, path, *, path_params=None, query_params=None):
        # Record the request as seen by the ingest layer.
        self.calls.append(
            {
                "method": method,
                "path": path,
                "path_params": dict(path_params or {}),
                "query_params": dict(query_params or {}),
            }
        )

        self._n += 1
        if self._n == 1:
            return {"attempt": 1}, _hr(status_code=422, ok=False, payload_json={"detail": "bad date"}), "MISS"
        return {"attempt": 2}, _hr(status_code=200, ok=True, payload_json={"data": {"ok": True}}), "MISS"


@pytest.mark.asyncio
async def test_market_tide_retries_once_without_date_on_422():
    client = _StubClient()

    market_call = PlannedCall(
        name="market_tide",
        method="GET",
        path="/api/market/market-tide",
        path_params={},
        query_params={"date": "{date}", "interval_5m": True},
        is_market=True,
        purpose="context-only",
        decision_path=False,
        missing_affects_confidence=False,
        stale_affects_confidence=False,
    )

    results = await fetch_all(
        client,
        tickers=[],
        date_str="2026-03-05",
        core=[],
        market=[market_call],
        max_concurrency=2,
    )

    # First request uses date; retry removes it.
    assert len(client.calls) == 2
    assert client.calls[0]["query_params"].get("date") == "2026-03-05"
    assert "date" not in client.calls[1]["query_params"]

    # Final stored result reflects the retry (status 200) and the date-less query params.
    assert len(results) == 1
    (_tkr, _call, _sig, qp_used, res, _cb) = results[0]
    assert "date" not in qp_used
    assert res.status_code == 200
