from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict


@dataclass(frozen=True)
class LogContext:
    service: str
    env: str = "local"


class JsonFormatter(logging.Formatter):
    def __init__(self, ctx: LogContext):
        super().__init__()
        self._ctx = ctx

    def format(self, record: logging.LogRecord) -> str:
        base: Dict[str, Any] = {
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "service": self._ctx.service,
            "env": self._ctx.env,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # common correlation ids
        for k in ("run_id", "ticker", "endpoint_id", "snapshot_id", "event_id", "prediction_id"):
            v = getattr(record, k, None)
            if v is not None:
                base[k] = v
        extra_json = getattr(record, "json", None)
        if isinstance(extra_json, dict):
            base.update(extra_json)
        if record.exc_info:
            base["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(base, ensure_ascii=False)


def configure_logging(service: str, level: str = "INFO") -> None:
    env = os.getenv("APP_ENV", "local")
    ctx = LogContext(service=service, env=env)

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter(ctx))
    root.addHandler(handler)
