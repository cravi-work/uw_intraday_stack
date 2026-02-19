from __future__ import annotations

import difflib
import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import duckdb
import yaml


UTC = timezone.utc


def _ensure_config_version(con: duckdb.DuckDBPyConnection, cfg_yaml_text: str) -> int:
    """Insert config into meta_config if new; return config_version.

    This is critical for auditability: adapter actions should reference a concrete,
    versioned config record in DuckDB, not just a file write on disk.
    """
    h = hashlib.sha256(cfg_yaml_text.encode("utf-8")).hexdigest()
    row = con.execute("SELECT config_version FROM meta_config WHERE config_hash = ?", [h]).fetchone()
    if row and row[0] is not None:
        return int(row[0])

    # Allocate a new version deterministically via DuckDB sequence.
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


@dataclass(frozen=True)
class AdaptResult:
    changed: bool
    reason: str


def compute_recent_metrics(con: duckdb.DuckDBPyConnection, window: int) -> Optional[Dict[str, float]]:
    row = con.execute(
        """SELECT avg(brier_score) AS brier, avg(log_loss) AS logloss
             FROM (
               SELECT brier_score, log_loss
               FROM predictions
               WHERE brier_score IS NOT NULL AND log_loss IS NOT NULL
               ORDER BY created_at_utc DESC
               LIMIT ?
             )""",
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
    metrics = compute_recent_metrics(con, window=window)
    if not metrics:
        return AdaptResult(False, "insufficient_metrics")

    brier = metrics["brier"]
    logloss = metrics["logloss"]

    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    changed = False
    diff: Dict[str, Any] = {"metrics": metrics, "changes": {}}

    # If severe drift => increase flat mass / reduce confidence cap (safe mode bias)
    if brier >= drift_bad["brier_bad"] or logloss >= drift_bad["logloss_bad"]:
        # reduce confidence cap (bounded)
        cap_bounds = bounds["confidence_cap"]
        cur_cap = float(cfg["model"]["confidence_cap"])
        new_cap = _step_towards(cur_cap, cap_bounds["min"], cap_bounds["max_step"])
        new_cap = _clamp(new_cap, cap_bounds["min"], cap_bounds["max"])
        if new_cap != cur_cap:
            cfg["model"]["confidence_cap"] = float(new_cap)
            diff["changes"]["confidence_cap"] = {"from": cur_cap, "to": new_cap}
            changed = True

        # increase flat_from_data_quality_scale (bounded)
        flat_bounds = bounds["flat_from_data_quality_scale"]
        cur_flat = float(cfg["model"]["flat_from_data_quality_scale"])
        new_flat = _step_towards(cur_flat, flat_bounds["max"], flat_bounds["max_step"])
        new_flat = _clamp(new_flat, flat_bounds["min"], flat_bounds["max"])
        if new_flat != cur_flat:
            cfg["model"]["flat_from_data_quality_scale"] = float(new_flat)
            diff["changes"]["flat_from_data_quality_scale"] = {"from": cur_flat, "to": new_flat}
            changed = True

        reason = "drift_bad"
    elif brier >= drift_warn["brier_warn"] or logloss >= drift_warn["logloss_warn"]:
        # mild drift => shrink weights toward 0 (bounded step)
        w_bounds = bounds["weights"]
        for k, w in cfg["model"]["weights"].items():
            w = float(w)
            target = 0.0
            new_w = _step_towards(w, target, w_bounds["max_step"])
            new_w = _clamp(new_w, w_bounds["min"], w_bounds["max"])
            if new_w != w:
                cfg["model"]["weights"][k] = float(new_w)
                diff["changes"].setdefault("weights", {})[k] = {"from": w, "to": new_w}
                changed = True
        reason = "drift_warn"
    else:
        return AdaptResult(False, "no_drift")

    if not changed:
        return AdaptResult(False, "bounded_no_change")

    # Write versioned config copy + update main config file
    p = Path(config_path)
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    versioned = p.with_name(f"config.{ts}.yaml")
    versioned.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    p.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    cfg_yaml_text = yaml.safe_dump(cfg, sort_keys=False)
    new_config_version = _ensure_config_version(con, cfg_yaml_text)

    # Insert into adapter_actions (audit)
    action_id = str(uuid.uuid4())
    con.execute(
        """INSERT INTO adapter_actions(action_id, run_id, from_config_version, to_config_version, reason, diff_json)
             VALUES (?, ?, ?, ?, ?, ?)""",
        [action_id, run_id, int(config_version_from_db), int(new_config_version), reason, json.dumps(diff, ensure_ascii=False)],
    )
    return AdaptResult(True, reason)
