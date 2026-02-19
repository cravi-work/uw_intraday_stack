from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


_ENV_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)(:-([^}]*))?\}")


def _expand_env(value: str) -> str:
    def repl(m: re.Match) -> str:
        var = m.group(1)
        default = m.group(3) if m.group(2) else ""
        return os.getenv(var, default)
    return _ENV_PATTERN.sub(repl, value)


def _walk_expand(obj: Any) -> Any:
    if isinstance(obj, str):
        return _expand_env(obj)
    if isinstance(obj, list):
        return [_walk_expand(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _walk_expand(v) for k, v in obj.items()}
    return obj


@dataclass(frozen=True)
class AppConfig:
    raw: Dict[str, Any]


def load_yaml(path: str | Path) -> AppConfig:
    p = Path(path)
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("config root must be a mapping")
    expanded = _walk_expand(data)
    return AppConfig(raw=expanded)


def load_endpoint_plan(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "plans" not in data:
        raise ValueError("endpoint_plan.yaml must contain a 'plans' mapping")
    return data
