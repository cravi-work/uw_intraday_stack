from __future__ import annotations

import json
import math
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional, Any, Dict, Tuple

import duckdb

from .models import HorizonKind, LabelContractSpec, PredictionTargetSpec, SignalState, predicted_class

UTC = timezone.utc
logger = logging.getLogger(__name__)


SESSION_BOUNDARY_TRUNCATE = "TRUNCATE_TO_SESSION_CLOSE"
SESSION_BOUNDARY_REQUIRE = "REQUIRE_TARGET_WITHIN_SESSION"


@dataclass(frozen=True)
class ResolvedValidationContract:
    target_spec: PredictionTargetSpec
    label_contract: LabelContractSpec

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_spec": self.target_spec.to_dict(),
            "label_contract": self.label_contract.to_dict(),
        }


@dataclass(frozen=True)
class ValidationResult:
    updated: int
    skipped: int
    leakage_violations: int = 0
    reason_counts: Dict[str, int] = field(default_factory=dict)


def _increment_reason(reason_counts: Dict[str, int], reason: str) -> None:
    reason_counts[reason] = int(reason_counts.get(reason, 0)) + 1


def _clamp_prob(p: float, eps: float = 1e-12) -> float:
    return max(eps, min(1.0 - eps, float(p)))


def _ensure_utc(dt_val: Optional[datetime]) -> Optional[datetime]:
    if dt_val is None:
        return None
    if dt_val.tzinfo is None:
        return dt_val.replace(tzinfo=UTC)
    return dt_val.astimezone(UTC)


def _is_valid_prob(p: Any) -> bool:
    return isinstance(p, (float, int)) and math.isfinite(p) and 0.0 <= p <= 1.0


def _is_valid_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and math.isfinite(v)


def _safe_json_loads(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
            return dict(decoded) if isinstance(decoded, dict) else {}
        except Exception:
            return {}
    return {}


def _coerce_horizon_kind(value: Any) -> Optional[HorizonKind]:
    if isinstance(value, HorizonKind):
        return value
    try:
        return HorizonKind(str(value).upper())
    except Exception:
        return None


def _parse_target_spec(raw: Any) -> Optional[PredictionTargetSpec]:
    if not isinstance(raw, dict):
        return None
    horizon_kind = _coerce_horizon_kind(raw.get("horizon_kind") or HorizonKind.FIXED)
    if horizon_kind is None:
        return None
    try:
        horizon_minutes = raw.get("horizon_minutes")
        return PredictionTargetSpec(
            target_name=str(raw.get("target_name") or ""),
            target_version=str(raw.get("target_version") or ""),
            class_labels=tuple(raw.get("class_labels") or ("UP", "DOWN", "FLAT")),  # type: ignore[arg-type]
            horizon_kind=horizon_kind,
            horizon_minutes=int(horizon_minutes) if horizon_minutes is not None else None,
            flat_threshold_pct=float(raw["flat_threshold_pct"]) if raw.get("flat_threshold_pct") is not None else None,
            probability_tolerance=float(raw.get("probability_tolerance", 1e-6)),
            contract_source=str(raw.get("contract_source") or "runtime_config"),
        )
    except (TypeError, ValueError):
        return None


def _parse_label_contract(raw: Any) -> Optional[LabelContractSpec]:
    if not isinstance(raw, dict):
        return None
    try:
        return LabelContractSpec(
            label_version=str(raw.get("label_version") or ""),
            session_boundary_rule=str(raw.get("session_boundary_rule") or ""),
            flat_threshold_pct=float(raw.get("flat_threshold_pct")),
            flat_threshold_policy=str(raw.get("flat_threshold_policy") or "ABS_RETURN_BAND"),
            threshold_policy_version=str(raw.get("threshold_policy_version") or raw.get("policy_version") or ""),
            neutral_threshold=float(raw["neutral_threshold"]) if raw.get("neutral_threshold") is not None else None,
            direction_margin=float(raw["direction_margin"]) if raw.get("direction_margin") is not None else None,
            contract_source=str(raw.get("contract_source") or "runtime_config"),
        )
    except (TypeError, ValueError):
        return None


def _resolve_validation_contract(meta_json: Dict[str, Any]) -> Tuple[Optional[ResolvedValidationContract], Optional[str]]:
    prediction_contract = meta_json.get("prediction_contract") or {}
    probability_contract = meta_json.get("probability_contract") or {}

    target_payload = prediction_contract.get("target_spec") or probability_contract.get("target_spec")
    target_spec = _parse_target_spec(target_payload)
    if target_spec is None:
        return None, "MISSING_TARGET_SPEC"
    if not target_spec.is_valid():
        return None, "INVALID_TARGET_SPEC"

    label_payload = prediction_contract.get("label_contract")
    if not isinstance(label_payload, dict):
        label_payload = {
            "label_version": prediction_contract.get("label_version"),
            "session_boundary_rule": prediction_contract.get("session_boundary_rule"),
            "flat_threshold_pct": prediction_contract.get("flat_threshold_pct", target_spec.flat_threshold_pct),
            "flat_threshold_policy": prediction_contract.get("flat_threshold_policy"),
            "threshold_policy_version": prediction_contract.get("threshold_policy_version"),
            "neutral_threshold": prediction_contract.get("neutral_threshold"),
            "direction_margin": prediction_contract.get("direction_margin"),
            "contract_source": prediction_contract.get("contract_source"),
        }
    label_contract = _parse_label_contract(label_payload)
    if label_contract is None:
        return None, "MISSING_LABEL_CONTRACT"
    if not label_contract.is_valid():
        return None, "INVALID_LABEL_CONTRACT"

    explicit_target_name = prediction_contract.get("target_name")
    explicit_target_version = prediction_contract.get("target_version")
    explicit_label_version = prediction_contract.get("label_version")
    if explicit_target_name is not None and str(explicit_target_name) != target_spec.target_name:
        return None, "TARGET_CONTRACT_MISMATCH"
    if explicit_target_version is not None and str(explicit_target_version) != target_spec.target_version:
        return None, "TARGET_CONTRACT_MISMATCH"
    if explicit_label_version is not None and str(explicit_label_version) != label_contract.label_version:
        return None, "LABEL_CONTRACT_MISMATCH"

    return ResolvedValidationContract(target_spec=target_spec, label_contract=label_contract), None


def _as_iso(dt_val: Optional[datetime]) -> Optional[str]:
    if dt_val is None:
        return None
    return _ensure_utc(dt_val).isoformat()


def _record_validation_outcome(
    con: duckdb.DuckDBPyConnection,
    prediction_id: str,
    meta_json: Dict[str, Any],
    *,
    status: str,
    reason: str,
    details: Optional[Dict[str, Any]] = None,
    contract: Optional[ResolvedValidationContract] = None,
) -> None:
    updated_meta = dict(meta_json or {})
    outcome_payload: Dict[str, Any] = {
        "status": status,
        "reason_code": reason,
    }
    if contract is not None:
        outcome_payload.update(
            {
                "target_name": contract.target_spec.target_name,
                "target_version": contract.target_spec.target_version,
                "label_version": contract.label_contract.label_version,
                "threshold_policy_version": contract.label_contract.threshold_policy_version,
                "session_boundary_rule": contract.label_contract.session_boundary_rule,
            }
        )
    if details:
        outcome_payload.update(details)
    updated_meta["validation_outcome"] = outcome_payload
    updated_meta["validation_error"] = reason if status == "SKIPPED" else None
    con.execute(
        "UPDATE predictions SET meta_json = ? WHERE prediction_id = ?",
        [json.dumps(updated_meta), prediction_id],
    )


def _record_validation_skip(
    con: duckdb.DuckDBPyConnection,
    prediction_id: str,
    meta_json: Dict[str, Any],
    reason: str,
    *,
    details: Optional[Dict[str, Any]] = None,
    contract: Optional[ResolvedValidationContract] = None,
) -> None:
    _record_validation_outcome(
        con,
        prediction_id,
        meta_json,
        status="SKIPPED",
        reason=reason,
        details=details,
        contract=contract,
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


def _resolve_target_timestamp(
    *,
    asof_ts_utc: datetime,
    horizon_kind: HorizonKind,
    horizon_minutes: Optional[int],
    horizon_seconds: Optional[int],
    market_close_utc: Optional[datetime],
    post_end_utc: Optional[datetime],
    session_label: Optional[str],
    is_early_close: bool,
    contract: ResolvedValidationContract,
) -> Tuple[Optional[datetime], Dict[str, Any], Optional[str]]:
    details: Dict[str, Any] = {
        "session_boundary_rule": contract.label_contract.session_boundary_rule,
        "session_label": session_label,
        "is_early_close": bool(is_early_close),
    }

    if horizon_kind == HorizonKind.TO_CLOSE:
        if not _is_valid_num(horizon_seconds) or int(horizon_seconds) <= 0:
            return None, details, "INVALID_TO_CLOSE_HORIZON_SECONDS"
        nominal_target_ts = asof_ts_utc + timedelta(seconds=int(horizon_seconds))
    elif horizon_kind == HorizonKind.FIXED:
        if not _is_valid_num(horizon_minutes) or int(horizon_minutes) <= 0:
            return None, details, "INVALID_FIXED_HORIZON_MINUTES"
        nominal_target_ts = asof_ts_utc + timedelta(minutes=int(horizon_minutes))
    else:
        return None, details, "INVALID_HORIZON_KIND"

    session_boundary_ts = None
    if str(session_label).upper() == "AFTERHOURS" and post_end_utc is not None:
        session_boundary_ts = post_end_utc
    elif market_close_utc is not None:
        session_boundary_ts = market_close_utc
    elif post_end_utc is not None and str(session_label).upper() == "AFTERHOURS":
        session_boundary_ts = post_end_utc

    details.update(
        {
            "nominal_target_ts_utc": _as_iso(nominal_target_ts),
            "session_boundary_ts_utc": _as_iso(session_boundary_ts),
            "session_boundary_truncated": False,
            "half_day_truncation": False,
        }
    )

    if (
        horizon_kind == HorizonKind.FIXED
        and session_boundary_ts is not None
        and nominal_target_ts > session_boundary_ts
    ):
        if contract.label_contract.session_boundary_rule == SESSION_BOUNDARY_TRUNCATE:
            details["session_boundary_truncated"] = True
            details["half_day_truncation"] = bool(is_early_close)
            details["truncation_reason"] = "HALF_DAY_CLOSE" if is_early_close else "SESSION_CLOSE"
            return session_boundary_ts, details, None
        if contract.label_contract.session_boundary_rule == SESSION_BOUNDARY_REQUIRE:
            details["session_boundary_violation"] = True
            return None, details, "TARGET_EXCEEDS_SESSION_BOUNDARY"

    return nominal_target_ts, details, None


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
    reason_counts: Dict[str, int] = {}

    rows = con.execute(
        """SELECT p.prediction_id, p.snapshot_id, p.horizon_kind, p.horizon_minutes, p.horizon_seconds,
                  p.start_price, p.prob_up, p.prob_down, p.prob_flat, s.ticker, s.asof_ts_utc, p.decision_state,
                  p.decision_window_id, p.validation_eligible, p.meta_json,
                  s.market_close_utc, s.post_end_utc, s.is_early_close, s.session_label
           FROM predictions p
           JOIN snapshots s ON s.snapshot_id = p.snapshot_id
           WHERE p.realized_at_utc IS NULL"""
    ).fetchall()

    updated = 0
    skipped = 0
    leakage_violations = 0

    cols_pred = [r[1] for r in con.execute("PRAGMA table_info('predictions')").fetchall()]
    has_outcome_price = "outcome_price" in cols_pred

    set_parts = [
        "realized_at_utc = ?",
        "outcome_label = ?",
        "brier_score = ?",
        "log_loss = ?",
        "outcome_realized = ?",
        "is_correct = ?",
        "meta_json = ?",
    ]
    if has_outcome_price:
        set_parts.insert(0, "outcome_price = ?")

    update_sql = "UPDATE predictions SET " + ", ".join(set_parts) + " WHERE prediction_id = ?"

    for (
        prediction_id,
        snapshot_id,
        horizon_kind_raw,
        horizon_minutes,
        horizon_seconds,
        start_price,
        prob_up,
        prob_down,
        prob_flat,
        ticker,
        asof_ts_utc,
        decision_state,
        window_id,
        is_eligible,
        meta_json_raw,
        market_close_utc,
        post_end_utc,
        is_early_close,
        session_label,
    ) in rows:
        pid = str(prediction_id)
        meta_json = _safe_json_loads(meta_json_raw)
        asof_ts_utc = _ensure_utc(asof_ts_utc)
        market_close_utc = _ensure_utc(market_close_utc)
        post_end_utc = _ensure_utc(post_end_utc)

        if asof_ts_utc is None:
            reason = "MISSING_ASOF_TIMESTAMP"
            _record_validation_skip(con, pid, meta_json, reason)
            _increment_reason(reason_counts, reason)
            skipped += 1
            continue

        contract, contract_error = _resolve_validation_contract(meta_json)
        if contract_error is not None or contract is None:
            reason = contract_error or "MISSING_VALIDATION_CONTRACT"
            details = {
                "asof_ts_utc": _as_iso(asof_ts_utc),
                "decision_window_id": window_id,
            }
            _record_validation_skip(con, pid, meta_json, reason, details=details)
            _increment_reason(reason_counts, reason)
            skipped += 1
            continue

        if abs(float(contract.label_contract.flat_threshold_pct) - float(flat_threshold_pct)) > 1e-12:
            reason = "FLAT_THRESHOLD_POLICY_MISMATCH"
            details = {
                "contract_flat_threshold_pct": float(contract.label_contract.flat_threshold_pct),
                "validator_flat_threshold_pct": float(flat_threshold_pct),
                "decision_window_id": window_id,
            }
            _record_validation_skip(con, pid, meta_json, reason, details=details, contract=contract)
            _increment_reason(reason_counts, reason)
            skipped += 1
            continue

        if not is_eligible:
            reason = "SKIP_INELIGIBLE"
            _record_validation_skip(
                con,
                pid,
                meta_json,
                reason,
                details={"decision_window_id": window_id},
                contract=contract,
            )
            _increment_reason(reason_counts, reason)
            skipped += 1
            logger.info(
                "Skipped prediction %s (ineligible)",
                pid,
                extra={"counter": "validation_rows_skipped_ineligible", "prediction_id": pid},
            )
            continue

        if decision_state == SignalState.NO_SIGNAL.value:
            reason = "SKIP_NO_SIGNAL"
            _record_validation_skip(con, pid, meta_json, reason, contract=contract)
            _increment_reason(reason_counts, reason)
            skipped += 1
            continue

        if prob_up is None or prob_down is None or prob_flat is None:
            reason = "SKIP_NULL_PROBS"
            _record_validation_skip(con, pid, meta_json, reason, contract=contract)
            _increment_reason(reason_counts, reason)
            skipped += 1
            continue

        valid_probs = _is_valid_prob(prob_up) and _is_valid_prob(prob_down) and _is_valid_prob(prob_flat)
        if valid_probs:
            prob_sum = float(prob_up) + float(prob_down) + float(prob_flat)
            if not (0.999 <= prob_sum <= 1.001):
                reason = "SKIP_BAD_PROB_SUM"
                _record_validation_skip(con, pid, meta_json, reason, contract=contract)
                _increment_reason(reason_counts, reason)
                skipped += 1
                continue
        else:
            reason = "SKIP_INVALID_PROBS"
            _record_validation_skip(con, pid, meta_json, reason, contract=contract)
            _increment_reason(reason_counts, reason)
            skipped += 1
            continue

        horizon_kind = _coerce_horizon_kind(horizon_kind_raw)
        if horizon_kind is None:
            reason = "INVALID_HORIZON_KIND"
            _record_validation_skip(con, pid, meta_json, reason, contract=contract)
            _increment_reason(reason_counts, reason)
            skipped += 1
            continue

        target_ts, target_details, target_reason = _resolve_target_timestamp(
            asof_ts_utc=asof_ts_utc,
            horizon_kind=horizon_kind,
            horizon_minutes=horizon_minutes,
            horizon_seconds=horizon_seconds,
            market_close_utc=market_close_utc,
            post_end_utc=post_end_utc,
            session_label=session_label,
            is_early_close=bool(is_early_close),
            contract=contract,
        )
        if target_reason is not None or target_ts is None:
            reason = target_reason or "INVALID_TARGET_TIMESTAMP"
            details = {
                **target_details,
                "asof_ts_utc": _as_iso(asof_ts_utc),
                "decision_window_id": window_id,
            }
            _record_validation_skip(con, pid, meta_json, reason, details=details, contract=contract)
            _increment_reason(reason_counts, reason)
            skipped += 1
            continue

        if target_ts > now_utc:
            continue

        max_drift_ts = target_ts + timedelta(minutes=int(max_horizon_drift_minutes))

        if start_price is not None:
            start_spot = float(start_price)
        else:
            start_spot_row = con.execute(
                "SELECT feature_value FROM features WHERE snapshot_id = ? AND feature_key = 'spot' LIMIT 1",
                [str(snapshot_id)],
            ).fetchone()
            if not start_spot_row or start_spot_row[0] is None:
                reason = "MISSING_START_PRICE"
                _record_validation_skip(
                    con,
                    pid,
                    meta_json,
                    reason,
                    details={**target_details, "target_ts_utc": _as_iso(target_ts)},
                    contract=contract,
                )
                _increment_reason(reason_counts, reason)
                skipped += 1
                continue
            start_spot = float(start_spot_row[0])

        if not _is_valid_num(start_spot) or start_spot <= 0:
            reason = "INVALID_START_PRICE"
            _record_validation_skip(
                con,
                pid,
                meta_json,
                reason,
                details={**target_details, "target_ts_utc": _as_iso(target_ts)},
                contract=contract,
            )
            _increment_reason(reason_counts, reason)
            skipped += 1
            continue

        realized_snap_row = con.execute(
            """SELECT snapshot_id, asof_ts_utc
               FROM snapshots
               WHERE ticker = ? AND asof_ts_utc >= ? AND asof_ts_utc > ?
               ORDER BY asof_ts_utc ASC LIMIT 1""",
            [ticker, target_ts, asof_ts_utc],
        ).fetchone()

        if not realized_snap_row:
            reason = "UNAVAILABLE_REALIZED_OUTCOME_NO_SNAPSHOT"
            _record_validation_skip(
                con,
                pid,
                meta_json,
                reason,
                details={**target_details, "target_ts_utc": _as_iso(target_ts)},
                contract=contract,
            )
            _increment_reason(reason_counts, reason)
            skipped += 1
            continue

        realized_snapshot_id, realized_asof = realized_snap_row
        realized_asof = _ensure_utc(realized_asof)

        if realized_asof is not None and realized_asof > max_drift_ts:
            reason = "TARGET_EXCEEDED_MAX_DRIFT"
            _record_validation_skip(
                con,
                pid,
                meta_json,
                reason,
                details={
                    **target_details,
                    "target_ts_utc": _as_iso(target_ts),
                    "realized_at_utc": _as_iso(realized_asof),
                    "max_drift_ts_utc": _as_iso(max_drift_ts),
                },
                contract=contract,
            )
            _increment_reason(reason_counts, reason)
            skipped += 1
            leakage_violations += 1
            logger.warning(
                "Leakage guard blocked prediction %s",
                pid,
                extra={"counter": "leakage_guard_violations", "prediction_id": pid, "window_id": window_id},
            )
            continue

        if realized_asof is None or (realized_asof - target_ts) > tol:
            reason = "UNAVAILABLE_REALIZED_OUTCOME_OUTSIDE_TOLERANCE"
            _record_validation_skip(
                con,
                pid,
                meta_json,
                reason,
                details={
                    **target_details,
                    "target_ts_utc": _as_iso(target_ts),
                    "realized_at_utc": _as_iso(realized_asof),
                    "tolerance_minutes": int(tolerance_minutes),
                },
                contract=contract,
            )
            _increment_reason(reason_counts, reason)
            skipped += 1
            continue

        realized_spot_row = con.execute(
            "SELECT feature_value FROM features WHERE snapshot_id = ? AND feature_key = 'spot' LIMIT 1",
            [str(realized_snapshot_id)],
        ).fetchone()
        if not realized_spot_row or realized_spot_row[0] is None:
            reason = "UNAVAILABLE_REALIZED_OUTCOME_MISSING_SPOT"
            _record_validation_skip(
                con,
                pid,
                meta_json,
                reason,
                details={
                    **target_details,
                    "target_ts_utc": _as_iso(target_ts),
                    "realized_at_utc": _as_iso(realized_asof),
                },
                contract=contract,
            )
            _increment_reason(reason_counts, reason)
            skipped += 1
            continue

        realized_spot = float(realized_spot_row[0])
        if not _is_valid_num(realized_spot) or realized_spot <= 0:
            reason = "INVALID_REALIZED_PRICE"
            _record_validation_skip(
                con,
                pid,
                meta_json,
                reason,
                details={
                    **target_details,
                    "target_ts_utc": _as_iso(target_ts),
                    "realized_at_utc": _as_iso(realized_asof),
                },
                contract=contract,
            )
            _increment_reason(reason_counts, reason)
            skipped += 1
            continue

        label = realized_label(start_spot, realized_spot, float(contract.label_contract.flat_threshold_pct))
        if label == "SKIPPED":
            reason = "INVALID_REALIZED_LABEL_RESULT"
            _record_validation_skip(
                con,
                pid,
                meta_json,
                reason,
                details={
                    **target_details,
                    "target_ts_utc": _as_iso(target_ts),
                    "realized_at_utc": _as_iso(realized_asof),
                },
                contract=contract,
            )
            _increment_reason(reason_counts, reason)
            skipped += 1
            continue

        pu = float(prob_up)
        pd = float(prob_down)
        pf = float(prob_flat)
        bs = brier_3class(pu, pd, pf, label)
        ll = logloss_3class(pu, pd, pf, label)
        pred_lbl = predicted_class(pu, pd, pf)

        updated_meta = dict(meta_json)
        updated_meta["validation_outcome"] = {
            "status": "UPDATED",
            "reason_code": "UPDATED_VALIDATED",
            "target_name": contract.target_spec.target_name,
            "target_version": contract.target_spec.target_version,
            "label_version": contract.label_contract.label_version,
            "threshold_policy_version": contract.label_contract.threshold_policy_version,
            "session_boundary_rule": contract.label_contract.session_boundary_rule,
            "target_ts_utc": _as_iso(target_ts),
            "realized_at_utc": _as_iso(realized_asof),
            "session_boundary_truncated": bool(target_details.get("session_boundary_truncated", False)),
            "half_day_truncation": bool(target_details.get("half_day_truncation", False)),
            "flat_threshold_pct": float(contract.label_contract.flat_threshold_pct),
        }
        updated_meta["validation_error"] = None

        params = []
        if has_outcome_price:
            params.append(realized_spot)

        params.extend(
            [
                realized_asof,
                label,
                bs,
                ll,
                True,
                bool(pred_lbl == label),
                json.dumps(updated_meta),
                pid,
            ]
        )

        con.execute(update_sql, params)
        updated += 1
        _increment_reason(reason_counts, "UPDATED_VALIDATED")

        logger.info(
            "Processed validation %s",
            pid,
            extra={"counter": "validation_rows_processed", "prediction_id": pid, "window_id": window_id},
        )

    return ValidationResult(
        updated=updated,
        skipped=skipped,
        leakage_violations=leakage_violations,
        reason_counts=reason_counts,
    )
