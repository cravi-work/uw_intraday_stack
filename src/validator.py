from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

import duckdb

from .models import predicted_class

UTC = timezone.utc

def _clamp_prob(p: float, eps: float = 1e-12) -> float:
    return max(eps, min(1.0 - eps, float(p)))

def _ensure_utc(dt_val: Optional[datetime]) -> Optional[datetime]:
    if dt_val is None:
        return None
    if dt_val.tzinfo is None:
        return dt_val.replace(tzinfo=UTC)
    return dt_val

def realized_label(start_price: float, realized_price: float, flat_threshold_pct: float) -> str:
    if start_price <= 0:
        return "SKIPPED"
    pct = (realized_price - start_price) / start_price
    if abs(pct) < flat_threshold_pct:
        return "FLAT"
    return "UP" if pct > 0 else "DOWN"

def brier_3class(prob_up: float, prob_down: float, prob_flat: float, label: str) -> float:
    o_up = 1.0 if label == "UP" else 0.0
    o_down = 1.0 if label == "DOWN" else 0.0
    o_flat = 1.0 if label == "FLAT" else 0.0
    return (prob_up - o_up) ** 2 + (prob_down - o_down) ** 2 + (prob_flat - o_flat) ** 2

def logloss_3class(prob_up: float, prob_down: float, prob_flat: float, label: str) -> float:
    p = {"UP": prob_up, "DOWN": prob_down, "FLAT": prob_flat}.get(label)
    if p is None:
        return float("nan")
    return -math.log(_clamp_prob(p))

@dataclass(frozen=True)
class ValidationResult:
    updated: int
    skipped: int

def validate_pending(
    con: duckdb.DuckDBPyConnection,
    *,
    now_utc: datetime,
    flat_threshold_pct: float,
    tolerance_minutes: int,
) -> ValidationResult:
    
    tol = timedelta(minutes=int(tolerance_minutes))
    now_utc = _ensure_utc(now_utc)

    rows = con.execute(
        """SELECT p.prediction_id, p.snapshot_id, p.horizon_kind, p.horizon_minutes, p.horizon_seconds, p.start_price, p.prob_up, p.prob_down, p.prob_flat,
                    s.ticker, s.asof_ts_utc
             FROM predictions p
             JOIN snapshots s ON s.snapshot_id = p.snapshot_id
             WHERE p.realized_at_utc IS NULL"""
    ).fetchall()

    updated = 0
    skipped = 0

    cols_pred = [r[1] for r in con.execute("PRAGMA table_info('predictions')").fetchall()]
    has_outcome_realized = "outcome_realized" in cols_pred
    has_is_correct = "is_correct" in cols_pred
    has_outcome_price = "outcome_price" in cols_pred

    set_parts = ["realized_at_utc = ?", "outcome_label = ?", "brier_score = ?", "log_loss = ?"]
    if has_outcome_price:
        set_parts.insert(0, "outcome_price = ?")
    if has_outcome_realized:
        set_parts.append("outcome_realized = ?")
    if has_is_correct:
        set_parts.append("is_correct = ?")
    update_sql = "UPDATE predictions SET " + ", ".join(set_parts) + " WHERE prediction_id = ?"

    for (prediction_id, snapshot_id, horizon_kind, horizon_minutes, horizon_seconds, start_price, prob_up, prob_down, prob_flat, ticker, asof_ts_utc) in rows:
        asof_ts_utc = _ensure_utc(asof_ts_utc)

        if asof_ts_utc is None:
            skipped += 1
            continue
        
        # Calculate exactly when this outcome is realized based on target type
        if horizon_kind == "TO_CLOSE":
            target_ts = asof_ts_utc + timedelta(seconds=int(horizon_seconds))
        else:
            target_ts = asof_ts_utc + timedelta(minutes=int(horizon_minutes))
            
        if target_ts > now_utc:
            continue

        if start_price is not None:
            start_spot = float(start_price)
        else:
            start_spot_row = con.execute("SELECT feature_value FROM features WHERE snapshot_id = ? AND feature_key = 'spot' LIMIT 1", [str(snapshot_id)]).fetchone()
            if not start_spot_row or start_spot_row[0] is None:
                skipped += 1
                continue
            start_spot = float(start_spot_row[0])

        realized_snap_row = con.execute(
            """SELECT snapshot_id, asof_ts_utc
                 FROM snapshots
                 WHERE ticker = ? AND asof_ts_utc >= ?
                 ORDER BY asof_ts_utc ASC
                 LIMIT 1""", [ticker, target_ts]).fetchone()
                 
        if not realized_snap_row:
            skipped += 1
            continue

        realized_snapshot_id, realized_asof = realized_snap_row
        realized_asof = _ensure_utc(realized_asof)

        if realized_asof is None or (realized_asof - target_ts) > tol:
            skipped += 1
            continue

        realized_spot_row = con.execute("SELECT feature_value FROM features WHERE snapshot_id = ? AND feature_key = 'spot' LIMIT 1", [str(realized_snapshot_id)]).fetchone()
        if not realized_spot_row or realized_spot_row[0] is None:
            skipped += 1
            continue
            
        realized_spot = float(realized_spot_row[0])

        label = realized_label(start_spot, realized_spot, float(flat_threshold_pct))
        if label == "SKIPPED":
            skipped += 1
            continue

        pu = float(prob_up) if prob_up is not None else 0.0
        pd = float(prob_down) if prob_down is not None else 0.0
        pf = float(prob_flat) if prob_flat is not None else 0.0
        bs = brier_3class(pu, pd, pf, label)
        ll = logloss_3class(pu, pd, pf, label)

        params = []
        if has_outcome_price:
            params.append(realized_spot)
        params.extend([realized_asof, label, bs, ll])
        if has_outcome_realized:
            params.append(True)
        if has_is_correct:
            pred_lbl = predicted_class(pu, pd, pf)
            params.append(bool(pred_lbl == label))
            
        params.append(str(prediction_id))

        con.execute(update_sql, params)
        updated += 1

    return ValidationResult(updated=updated, skipped=skipped)