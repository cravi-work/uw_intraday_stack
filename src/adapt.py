from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import timezone
from typing import Any, Dict, Mapping, Optional

import duckdb


UTC = timezone.utc

ADAPT_UNSUPPORTED_REASON = (
    "Adaptive runtime is disabled under the current contract. "
    "The repo does not provide a contract-safe live mutation path for model weights/calibration, "
    "and startup must reject adapt.enabled=true until a fully versioned schema/runtime is implemented."
)


class AdaptUnsupportedError(RuntimeError):
    """Raised when the disabled adaptive runtime is invoked."""


@dataclass(frozen=True)
class AdaptResult:
    changed: bool
    reason: str


@dataclass(frozen=True)
class AdaptSupportStatus:
    enabled_requested: bool
    supported: bool
    reason: str


@dataclass(frozen=True)
class AdaptConfigValidationResult:
    enabled: bool
    supported: bool
    rejection_reason: Optional[str]


def _ensure_config_version(con: duckdb.DuckDBPyConnection, cfg_yaml_text: str) -> int:
    """Insert config into meta_config if new; return config_version."""
    h = hashlib.sha256(cfg_yaml_text.encode("utf-8")).hexdigest()
    row = con.execute("SELECT config_version FROM meta_config WHERE config_hash = ?", [h]).fetchone()
    if row and row[0] is not None:
        return int(row[0])

    v = con.execute("SELECT nextval('seq_config_version')").fetchone()[0]
    con.execute(
        "INSERT INTO meta_config(config_version, config_hash, config_yaml) VALUES (?, ?, ?)",
        [int(v), h, cfg_yaml_text],
    )
    return int(v)



def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))



def _step_towards(x: float, target: float, max_step: float) -> float:
    if target > x:
        return min(x + max_step, target)
    return max(x - max_step, target)



def get_adapt_support_status(cfg: Optional[Mapping[str, Any]] = None) -> AdaptSupportStatus:
    adapt_cfg: Mapping[str, Any] = {}
    if isinstance(cfg, Mapping):
        raw = cfg.get("adapt")
        if isinstance(raw, Mapping):
            adapt_cfg = raw
    enabled_requested = bool(adapt_cfg.get("enabled", False))
    return AdaptSupportStatus(
        enabled_requested=enabled_requested,
        supported=False,
        reason=ADAPT_UNSUPPORTED_REASON,
    )



def validate_adapt_config(cfg: Optional[Mapping[str, Any]]) -> AdaptConfigValidationResult:
    status = get_adapt_support_status(cfg)
    if status.enabled_requested:
        raise ValueError(
            "adapt.enabled=true is not supported by the current runtime contract; "
            "adaptive writes are disabled until a fully versioned schema/runtime is implemented"
        )
    return AdaptConfigValidationResult(
        enabled=False,
        supported=status.supported,
        rejection_reason=status.reason,
    )



def compute_recent_metrics(con: duckdb.DuckDBPyConnection, window: int) -> Optional[Dict[str, float]]:
    """Return recent realized validation metrics using the current predictions schema.

    This helper remains query-safe for diagnostics even though adaptive mutation is disabled.
    """
    row = con.execute(
        """
        SELECT avg(brier_score) AS brier, avg(log_loss) AS logloss
        FROM (
          SELECT brier_score, log_loss
          FROM predictions
          WHERE brier_score IS NOT NULL AND log_loss IS NOT NULL
          ORDER BY COALESCE(realized_at_utc, source_ts_max_utc, source_ts_min_utc) DESC NULLS LAST,
                   prediction_business_key DESC NULLS LAST
          LIMIT ?
        )
        """,
        [int(window)],
    ).fetchone()
    if not row or row[0] is None or row[1] is None:
        return None
    return {"brier": float(row[0]), "logloss": float(row[1])}



def adapt_config_if_needed(
    con: duckdb.DuckDBPyConnection,
    *,
    run_id: str,
    config_path: str,
    config_version_from_db: int,
    window: int,
    drift_warn: Dict[str, float],
    drift_bad: Dict[str, float],
    bounds: Dict[str, Any],
) -> AdaptResult:
    """Disabled-by-contract adaptive path.

    The runtime intentionally rejects adaptive config mutation because it is not covered by the
    current schema/runtime contract. Callers must keep adapt.enabled=false until a future ticket
    supplies the full versioned write/audit path.
    """
    raise AdaptUnsupportedError(
        "Adaptive runtime is disabled under the current contract; "
        "config mutation and adapter audit writes are intentionally blocked"
    )
