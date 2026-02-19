from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import yaml


@dataclass(frozen=True)
class EndpointParam:
    name: str
    location: str
    required: bool
    schema_type: Optional[str] = None


@dataclass(frozen=True)
class EndpointDef:
    method: str
    path: str
    operation_id: Optional[str]
    summary: Optional[str]
    description: Optional[str]
    parameters: Tuple[EndpointParam, ...]
    category: str


class ApiCatalogError(RuntimeError):
    pass


class EndpointRegistry:
    """Hard allow-list for HTTP calls, loaded from api_catalog.generated.yaml."""

    def __init__(self, endpoints: List[EndpointDef], catalog_hash: str, catalog_source: str, version: int):
        self._endpoints = endpoints
        self._by_key: Dict[Tuple[str, str], EndpointDef] = {(e.method.upper(), e.path): e for e in endpoints}
        self.catalog_hash = catalog_hash
        self.catalog_source = catalog_source
        self.version = version

    def has(self, method: str, path: str) -> bool:
        return (method.upper(), path) in self._by_key

    def get(self, method: str, path: str) -> EndpointDef:
        k = (method.upper(), path)
        if k not in self._by_key:
            raise ApiCatalogError(f"Endpoint not in catalog: {method} {path}")
        return self._by_key[k]

    def allowed_query_params(self, method: str, path: str) -> List[str]:
        ep = self.get(method, path)
        return [p.name for p in ep.parameters if p.location == "query"]

    def validate_query_params(self, method: str, path: str, params: Mapping[str, Any]) -> None:
        allowed = set(self.allowed_query_params(method, path))
        unknown = [k for k in params.keys() if k not in allowed]
        if unknown:
            raise ApiCatalogError(f"Unknown query params for {method} {path}: {unknown}. Allowed={sorted(allowed)}")

    @staticmethod
    def normalize_params(params: Mapping[str, Any]) -> Dict[str, Any]:
        # stable + deterministic
        return {k: params[k] for k in sorted(params.keys()) if params[k] is not None}

    @staticmethod
    def params_hash(params: Mapping[str, Any]) -> str:
        norm = EndpointRegistry.normalize_params(params)
        b = json.dumps(norm, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(b).hexdigest()

    @staticmethod
    def signature(method: str, path: str, params: Mapping[str, Any]) -> str:
        norm = EndpointRegistry.normalize_params(params)
        return f"{method.upper()} {path} {json.dumps(norm, sort_keys=True, separators=(',', ':'), ensure_ascii=False)}"


def load_api_catalog(path: str | Path) -> EndpointRegistry:
    p = Path(path)
    if not p.exists():
        raise ApiCatalogError(f"Catalog file not found: {p}")

    raw = p.read_bytes()
    catalog_hash = hashlib.sha256(raw).hexdigest()

    data = yaml.safe_load(raw)
    if not isinstance(data, dict):
        raise ApiCatalogError("Catalog root must be a mapping")

    version = int(data.get("version", 0))
    source = str(data.get("source", ""))

    cats = data.get("categories")
    if not isinstance(cats, dict):
        raise ApiCatalogError("Catalog 'categories' must be a mapping")

    endpoints: List[EndpointDef] = []
    for cat_name, cat_obj in cats.items():
        eps = cat_obj.get("endpoints", [])
        if not isinstance(eps, list):
            raise ApiCatalogError(f"Category {cat_name} endpoints must be a list")
        for ep in eps:
            method = str(ep.get("method", "")).upper()
            path_s = str(ep.get("path", ""))
            op_id = ep.get("operationId")
            summary = ep.get("summary")
            desc = ep.get("description")
            params_list = ep.get("parameters", []) or []
            params: List[EndpointParam] = []
            if isinstance(params_list, list):
                for prm in params_list:
                    if not isinstance(prm, dict):
                        continue
                    params.append(
                        EndpointParam(
                            name=str(prm.get("name", "")),
                            location=str(prm.get("in", prm.get("location", "query"))),
                            required=bool(prm.get("required", False)),
                            schema_type=(prm.get("schema") or {}).get("type") if isinstance(prm.get("schema"), dict) else None,
                        )
                    )
            endpoints.append(
                EndpointDef(
                    method=method,
                    path=path_s,
                    operation_id=str(op_id) if op_id is not None else None,
                    summary=str(summary) if summary is not None else None,
                    description=str(desc) if desc is not None else None,
                    parameters=tuple(params),
                    category=str(cat_name),
                )
            )

    endpoints.sort(key=lambda e: (e.method, e.path, e.operation_id or ""))
    return EndpointRegistry(endpoints=endpoints, catalog_hash=catalog_hash, catalog_source=source, version=version)
