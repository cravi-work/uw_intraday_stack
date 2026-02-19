from __future__ import annotations
import datetime
from typing import Any

UTC = datetime.timezone.utc

def to_utc_dt(x: Any, *, fallback: datetime.datetime) -> datetime.datetime:
    """Deterministically normalizes any timestamp to a UTC aware datetime."""
    if x is None:
        return fallback
    if isinstance(x, (int, float)):
        return datetime.datetime.fromtimestamp(x, UTC)
    if isinstance(x, datetime.datetime):
        if x.tzinfo is None:
            return x.replace(tzinfo=UTC)
        return x.astimezone(UTC)
    return fallback