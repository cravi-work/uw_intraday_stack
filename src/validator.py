# src/validator.py
from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional, Any

import duckdb

from .models import predicted_class, SignalState

UTC = timezone.utc
logger = logging.getLogger(__name__)


def _clamp_prob(p: float, eps: float = 1e-12) -> float:
    return max(eps, min(1.0 - eps, float(p)))


def _ensure_utc(dt_val: Optional[datetime]) -> Optional[datetime]:
    if dt_val is None:
        return None
    if dt_val.tzinfo is None:
        return dt_val.replace(tzinfo=UTC)
    return dt_val


def _is_valid_prob(p: Any) -> bool:
    return isinstance(p, (float, int)) and math.isfinite(p) and 0.0 <= p <= 1.0


def _is_valid_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and math.isfinite(v)


def _record_validation_skip(con: duckdb.DuckDBPyConnection, prediction_id: str, reason: str) -> None:
    con.execute(
        "UPDATE predictions SET meta_json = json_insert(COALESCE(meta_json, '{}'), '$.validation_error', ?) WHERE prediction_id = ?",
        [reason, prediction_id]
    )


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
    leakage_violations: int = 0


# Task 12: No silent defaults in validation core signature
def validate_pending(
    con: duckdb.DuckDBPyConnection,
    *,
    now_utc: datetime,
    flat_threshold_pct: float,
    tolerance_minutes: int,
    max_horizon_drift_minutes: int,
) -> ValidationResult:
    
    tol = timedelta(minutes=int(tolerance_minutes))
    now_utc = _ensure_utc(now_utc)

    # CL-06: Added window_id and validation_eligible to selection. Removed eligibility filter for python-side tracking.
    rows = con.execute(
        f"""SELECT p.prediction_id, p.snapshot_id, p.horizon_kind, p.horizon_minutes, p.horizon_seconds, 
                  p.start_price, p.prob_up, p.prob_down, p.prob_flat, s.ticker, s.asof_ts_utc, p.decision_state,
                  p.decision_window_id, p.validation_eligible
           FROM predictions p 
           JOIN snapshots s ON s.snapshot_id = p.snapshot_id
           WHERE p.realized_at_utc IS NULL"""
    ).fetchall()

    updated = 0
    skipped = 0
    leakage_violations = 0

    cols_pred = [r[1] for r in con.execute("PRAGMA table_info('predictions')").fetchall()]
    has_outcome_price = "outcome_price" in cols_pred

    set_parts = ["realized_at_utc = ?", "outcome_label = ?", "brier_score = ?", "log_loss = ?", "outcome_realized = ?", "is_correct = ?"]
    if has_outcome_price:
        set_parts.insert(0, "outcome_price = ?")
        
    update_sql = "UPDATE predictions SET " + ", ".join(set_parts) + " WHERE prediction_id = ?"

    for (prediction_id, snapshot_id, horizon_kind, horizon_minutes, horizon_seconds, start_price, prob_up, prob_down, prob_flat, ticker, asof_ts_utc, decision_state, window_id, is_eligible) in rows:
        pid = str(prediction_id)
        asof_ts_utc = _ensure_utc(asof_ts_utc)
        
        if asof_ts_utc is None:
            skipped += 1
            continue

        # CL-06 Tracking logic for ineligible skips
        if not is_eligible:
            _record_validation_skip(con, pid, "SKIP_INELIGIBLE")
            skipped += 1
            logger.info(f"Skipped prediction {pid} (ineligible)", extra={"counter": "validation_rows_skipped_ineligible", "prediction_id": pid})
            continue

        if decision_state == SignalState.NO_SIGNAL.value:
            _record_validation_skip(con, pid, "SKIP_NO_SIGNAL")
            skipped += 1
            continue
        
        # STRICT VECTOR VALIDATION
        if prob_up is None or prob_down is None or prob_flat is None:
            _record_validation_skip(con, pid, "SKIP_NULL_PROBS")
            skipped += 1
            continue

        valid_probs = _is_valid_prob(prob_up) and _is_valid_prob(prob_down) and _is_valid_prob(prob_flat)
        if valid_probs:
            prob_sum = float(prob_up) + float(prob_down) + float(prob_flat)
            if not (0.999 <= prob_sum <= 1.001):
                _record_validation_skip(con, pid, "SKIP_BAD_PROB_SUM")
                skipped += 1
                continue
        else:
            _record_validation_skip(con, pid, "validation_skipped_invalid_probs")
            skipped += 1
            continue

        if horizon_kind not in ["FIXED", "TO_CLOSE"]:
            _record_validation_skip(con, pid, "invalid_horizon_kind")
            skipped += 1
            continue

        if horizon_kind == "TO_CLOSE":
            if not _is_valid_num(horizon_seconds) or horizon_seconds <= 0:
                _record_validation_skip(con, pid, "invalid_to_close_horizon_seconds")
                skipped += 1
                continue
            target_ts = asof_ts_utc + timedelta(seconds=int(horizon_seconds))
        else:
            if not _is_valid_num(horizon_minutes) or horizon_minutes <= 0:
                _record_validation_skip(con, pid, "invalid_fixed_horizon_minutes")
                skipped += 1
                continue
            target_ts = asof_ts_utc + timedelta(minutes=int(horizon_minutes))
            
        if target_ts > now_utc:
            continue

        max_drift_ts = target_ts + timedelta(minutes=int(max_horizon_drift_minutes))

        if start_price is not None:
            start_spot = float(start_price)
        else:
            start_spot_row = con.execute(
                "SELECT feature_value FROM features WHERE snapshot_id = ? AND feature_key = 'spot' LIMIT 1",
                [str(snapshot_id)]
            ).fetchone()
            
            if not start_spot_row or start_spot_row[0] is None:
                _record_validation_skip(con, pid, "missing_start_price")
                skipped += 1
                continue
                
            start_spot = float(start_spot_row[0])

        if not _is_valid_num(start_spot) or start_spot <= 0:
            _record_validation_skip(con, pid, "invalid_start_price")
            skipped += 1
            continue

        # LEAKAGE-FREE REALIZED PRICE TARGETING
        realized_snap_row = con.execute(
            """SELECT snapshot_id, asof_ts_utc 
               FROM snapshots
               WHERE ticker = ? AND asof_ts_utc >= ? AND asof_ts_utc > ?
               ORDER BY asof_ts_utc ASC LIMIT 1""", 
            [ticker, target_ts, asof_ts_utc]
        ).fetchone()
                 
        if not realized_snap_row:
            _record_validation_skip(con, pid, "no_realized_snapshot")
            skipped += 1
            continue

        realized_snapshot_id, realized_asof = realized_snap_row
        realized_asof = _ensure_utc(realized_asof)

        # CL-06 Leakage Rejection
        if realized_asof is not None and realized_asof > max_drift_ts:
            _record_validation_skip(con, pid, "target_exceeded_max_drift")
            skipped += 1
            leakage_violations += 1
            logger.warning(f"Leakage guard blocked prediction {pid}", extra={"counter": "leakage_guard_violations", "prediction_id": pid, "window_id": window_id})
            continue

        if realized_asof is None or (realized_asof - target_ts) > tol:
            _record_validation_skip(con, pid, "realized_snapshot_outside_tolerance")
            skipped += 1
            continue

        realized_spot_row = con.execute(
            "SELECT feature_value FROM features WHERE snapshot_id = ? AND feature_key = 'spot' LIMIT 1",
            [str(realized_snapshot_id)]
        ).fetchone()
        
        if not realized_spot_row or realized_spot_row[0] is None:
            _record_validation_skip(con, pid, "missing_realized_spot")
            skipped += 1
            continue
            
        realized_spot = float(realized_spot_row[0])
        
        if not _is_valid_num(realized_spot) or realized_spot <= 0:
            _record_validation_skip(con, pid, "invalid_realized_price")
            skipped += 1
            continue

        label = realized_label(start_spot, realized_spot, float(flat_threshold_pct))
        if label == "SKIPPED":
            _record_validation_skip(con, pid, "invalid_realized_label_result")
            skipped += 1
            continue

        pu = float(prob_up)
        pd = float(prob_down)
        pf = float(prob_flat)
        bs = brier_3class(pu, pd, pf, label)
        ll = logloss_3class(pu, pd, pf, label)

        params = []
        if has_outcome_price:
            params.append(realized_spot)
            
        pred_lbl = predicted_class(pu, pd, pf)
        params.extend([realized_asof, label, bs, ll, True, bool(pred_lbl == label), pid])

        con.execute(update_sql, params)
        updated += 1
        
        # CL-06 Validation Processing Telemetry
        logger.info(f"Processed validation {pid}", extra={"counter": "validation_rows_processed", "prediction_id": pid, "window_id": window_id})

    return ValidationResult(updated=updated, skipped=skipped, leakage_violations=leakage_violations)