from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple, Union, List

import httpx
from aiolimiter import AsyncLimiter

from .api_catalog_loader import ApiCatalogError, EndpointRegistry

logger = logging.getLogger(__name__)


class UwClientError(RuntimeError):
    pass


@dataclass(frozen=True)
class HttpResult:
    ok: bool
    status_code: Optional[int]
    payload_json: Optional[Any]
    payload_text: Optional[str]
    payload_hash: Optional[str]
    requested_at_utc: float
    received_at_utc: Optional[float]
    latency_ms: Optional[int]
    error_type: Optional[str]
    error_message: Optional[str]
    retry_count: int


class CircuitBreaker:
    def __init__(self, failure_threshold: int, cool_down_seconds: int, half_open_max_calls: int):
        self.failure_threshold = int(failure_threshold)
        self.cool_down_seconds = int(cool_down_seconds)
        self.half_open_max_calls = int(half_open_max_calls)
        self._state: Dict[str, Dict[str, Any]] = {}

    def _now(self) -> float:
        return time.time()

    def allow(self, key: str) -> bool:
        st = self._state.get(key)
        if not st:
            return True
        state = st.get("state", "CLOSED")
        if state == "CLOSED":
            return True
        if state == "OPEN":
            if self._now() >= float(st.get("open_until", 0)):
                st["state"] = "HALF_OPEN"
                st["half_open_calls"] = 0
                return True
            return False
        if state == "HALF_OPEN":
            return int(st.get("half_open_calls", 0)) < self.half_open_max_calls
        return True

    def on_success(self, key: str) -> None:
        self._state[key] = {"state": "CLOSED", "failures": 0, "open_until": 0, "half_open_calls": 0}

    def on_failure(self, key: str) -> None:
        st = self._state.setdefault(key, {"state": "CLOSED", "failures": 0, "open_until": 0, "half_open_calls": 0})
        st["failures"] = int(st.get("failures", 0)) + 1
        if st["state"] == "HALF_OPEN":
            st["half_open_calls"] = int(st.get("half_open_calls", 0)) + 1
        if int(st["failures"]) >= self.failure_threshold:
            st["state"] = "OPEN"
            st["open_until"] = self._now() + self.cool_down_seconds

    def snapshot_state(self, key: str) -> Dict[str, Any]:
        return dict(self._state.get(key, {"state": "CLOSED", "failures": 0, "open_until": 0, "half_open_calls": 0}))


class UwClient:
    def __init__(
        self,
        registry: EndpointRegistry,
        base_url: str,
        api_key_env: str,
        timeout_seconds: float,
        max_retries: int,
        backoff_seconds: Union[float, List[float]],
        max_concurrent_requests: int,
        rate_limit_per_second: int,
        circuit_failure_threshold: int,
        circuit_cool_down_seconds: int,
        circuit_half_open_max_calls: int,
    ):
        self.registry = registry
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.timeout_seconds = float(timeout_seconds)
        self.max_retries = int(max_retries)
        
        # Robustly handle scalar or list configurations
        if isinstance(backoff_seconds, (int, float)):
            self.backoff_seconds = [float(backoff_seconds)]
        else:
            self.backoff_seconds = [float(x) for x in backoff_seconds]
            
        self._limiter = AsyncLimiter(int(rate_limit_per_second), time_period=1.0)
        self._sem = asyncio.Semaphore(int(max_concurrent_requests))
        self._cb = CircuitBreaker(
            failure_threshold=circuit_failure_threshold,
            cool_down_seconds=circuit_cool_down_seconds,
            half_open_max_calls=circuit_half_open_max_calls,
        )
        self._client: Optional[httpx.AsyncClient] = None

    def _auth_headers(self) -> Dict[str, str]:
        api_key = os.getenv(self.api_key_env, "").strip()
        if not api_key:
            raise UwClientError(f"Missing API key env var: {self.api_key_env}")
        return {"Authorization": f"Bearer {api_key}"}

    async def __aenter__(self) -> "UwClient":
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(self.timeout_seconds),
            headers=self._auth_headers(),
        )
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._client:
            await self._client.aclose()
        self._client = None

    @staticmethod
    def _hash_payload(payload: Any) -> str:
        b = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(b).hexdigest()

    def _backoff(self, attempt: int) -> float:
        if attempt < len(self.backoff_seconds):
            return float(self.backoff_seconds[attempt])
        tail = self.backoff_seconds[-1] if self.backoff_seconds else 1.0
        return float(tail * (2 ** max(0, attempt - len(self.backoff_seconds) + 1)))

    async def request(
        self,
        method: str,
        path: str,
        *,
        path_params: Optional[Mapping[str, Any]] = None,
        query_params: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[str, HttpResult, Dict[str, Any]]:
        if self._client is None:
            raise UwClientError("Client not initialized; use 'async with UwClient(...)'")

        method_u = method.upper()
        self.registry.get(method_u, path)  # validate endpoint exists

        qp = dict(query_params or {})
        self.registry.validate_query_params(method_u, path, qp)
        endpoint_sig = self.registry.signature(method_u, path, qp)

        if not self._cb.allow(endpoint_sig):
            return endpoint_sig, HttpResult(
                ok=False,
                status_code=None,
                payload_json=None,
                payload_text=None,
                payload_hash=None,
                requested_at_utc=time.time(),
                received_at_utc=None,
                latency_ms=None,
                error_type="CircuitOpen",
                error_message="Circuit breaker open",
                retry_count=0,
            ), self._cb.snapshot_state(endpoint_sig)

        url_path = path
        if path_params:
            for k, v in path_params.items():
                url_path = url_path.replace("{" + str(k) + "}", str(v))

        last_err: Optional[HttpResult] = None
        for attempt in range(self.max_retries + 1):
            requested_at = time.time()
            try:
                async with self._sem:
                    async with self._limiter:
                        resp = await self._client.request(method_u, url_path, params=qp)
                received_at = time.time()
                latency_ms = int((received_at - requested_at) * 1000)

                if 200 <= resp.status_code < 300:
                    try:
                        payload_json = resp.json()
                        payload_hash = self._hash_payload(payload_json)
                    except Exception as je:
                        last_err = HttpResult(
                            ok=False, status_code=resp.status_code,
                            payload_json=None, payload_text=resp.text, payload_hash=None,
                            requested_at_utc=requested_at, received_at_utc=received_at,
                            latency_ms=latency_ms, error_type="JsonDecodeError", error_message=str(je),
                            retry_count=attempt,
                        )
                        self._cb.on_failure(endpoint_sig)
                        break

                    self._cb.on_success(endpoint_sig)
                    return endpoint_sig, HttpResult(
                        ok=True, status_code=resp.status_code,
                        payload_json=payload_json, payload_text=None, payload_hash=payload_hash,
                        requested_at_utc=requested_at, received_at_utc=received_at, latency_ms=latency_ms,
                        error_type=None, error_message=None, retry_count=attempt,
                    ), self._cb.snapshot_state(endpoint_sig)

                last_err = HttpResult(
                    ok=False, status_code=resp.status_code,
                    payload_json=None, payload_text=resp.text, payload_hash=None,
                    requested_at_utc=requested_at, received_at_utc=received_at, latency_ms=latency_ms,
                    error_type="HttpStatusError", error_message=f"HTTP {resp.status_code}", retry_count=attempt,
                )
                self._cb.on_failure(endpoint_sig)
                if resp.status_code >= 500 and attempt < self.max_retries:
                    await asyncio.sleep(self._backoff(attempt))
                    continue
                break

            except (httpx.TimeoutException, httpx.NetworkError) as ne:
                received_at = time.time()
                latency_ms = int((received_at - requested_at) * 1000)
                last_err = HttpResult(
                    ok=False, status_code=None,
                    payload_json=None, payload_text=None, payload_hash=None,
                    requested_at_utc=requested_at, received_at_utc=received_at, latency_ms=latency_ms,
                    error_type=type(ne).__name__, error_message=str(ne), retry_count=attempt,
                )
                self._cb.on_failure(endpoint_sig)
                if attempt < self.max_retries:
                    await asyncio.sleep(self._backoff(attempt))
                    continue
                break
            except ApiCatalogError:
                raise
            except Exception as e:
                received_at = time.time()
                latency_ms = int((received_at - requested_at) * 1000)
                last_err = HttpResult(
                    ok=False, status_code=None,
                    payload_json=None, payload_text=None, payload_hash=None,
                    requested_at_utc=requested_at, received_at_utc=received_at, latency_ms=latency_ms,
                    error_type=type(e).__name__, error_message=str(e), retry_count=attempt,
                )
                self._cb.on_failure(endpoint_sig)
                break

        assert last_err is not None
        return endpoint_sig, last_err, self._cb.snapshot_state(endpoint_sig)