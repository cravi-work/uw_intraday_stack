from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, is_dataclass
from datetime import date, datetime, time, timezone
from enum import Enum
from typing import Any, Dict, Mapping, MutableMapping, Optional


@dataclass(frozen=True)
class LogContext:
    service: str
    env: str = "local"


_STANDARD_LOG_RECORD_ATTRS = frozenset(logging.makeLogRecord({}).__dict__.keys()) | {"message", "asctime"}


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if value != value:
            return "NaN"
        if value == float("inf"):
            return "Infinity"
        if value == float("-inf"):
            return "-Infinity"
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (datetime, date, time)):
        if isinstance(value, datetime) and value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc).isoformat()
        return value.isoformat()
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(v) for v in value]
    if isinstance(value, BaseException):
        return {"type": value.__class__.__name__, "msg": str(value)}
    return str(value)


class JsonFormatter(logging.Formatter):
    def __init__(self, ctx: LogContext):
        super().__init__()
        self._ctx = ctx

    def _extra_fields(self, record: logging.LogRecord) -> Dict[str, Any]:
        direct_fields: Dict[str, Any] = {}
        json_fields: Dict[str, Any] = {}

        raw_json = getattr(record, "json", None)
        if isinstance(raw_json, Mapping):
            json_fields = _json_safe(raw_json)
            if not isinstance(json_fields, dict):
                json_fields = {"payload": json_fields}
        elif raw_json is not None:
            json_fields = {"payload": _json_safe(raw_json)}

        for key, value in record.__dict__.items():
            if key in _STANDARD_LOG_RECORD_ATTRS or key == "json" or key.startswith("_"):
                continue
            direct_fields[key] = _json_safe(value)

        merged: Dict[str, Any] = {}
        merged.update(json_fields)
        merged.update(direct_fields)
        return merged

    def format(self, record: logging.LogRecord) -> str:
        base: Dict[str, Any] = {
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "service": self._ctx.service,
            "env": self._ctx.env,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        base.update(self._extra_fields(record))

        if "event" not in base:
            counter_name = base.get("counter")
            if isinstance(counter_name, str) and counter_name:
                base["event"] = counter_name

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


def structured_log(
    logger: logging.Logger,
    level: int,
    *,
    event: str,
    msg: Optional[str] = None,
    counter: Optional[str] = None,
    **fields: Any,
) -> None:
    payload = {key: _json_safe(value) for key, value in fields.items() if value is not None}
    extra: Dict[str, Any] = {"event": event, "json": payload}
    if counter is not None:
        extra["counter"] = counter
    logger.log(level, msg or event, extra=extra)


def _mapping(raw: Any) -> Dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, Mapping):
        return dict(raw)
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except Exception:
            return {}
        return dict(parsed) if isinstance(parsed, Mapping) else {}
    return {}


def _listish(raw: Any) -> list[Any]:
    if raw is None:
        return []
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except Exception:
            return [raw]
        return list(parsed) if isinstance(parsed, list) else [parsed]
    if isinstance(raw, (list, tuple, set, frozenset)):
        return list(raw)
    return [raw]


def build_prediction_trace(prediction: Mapping[str, Any]) -> Dict[str, Any]:
    pred = dict(prediction)
    meta_json = _mapping(pred.get("meta_json"))
    probability_contract = _mapping(meta_json.get("probability_contract"))
    prediction_contract = _mapping(meta_json.get("prediction_contract"))
    target_spec = _mapping(prediction_contract.get("target_spec") or probability_contract.get("target_spec"))
    label_contract = _mapping(prediction_contract.get("label_contract"))
    calibration_ref = _mapping(probability_contract.get("calibration_artifact_ref"))
    horizon_contract = _mapping(meta_json.get("horizon_contract"))
    freshness_diag = _mapping(meta_json.get("freshness_registry_diagnostics"))
    alignment_diag = _mapping(meta_json.get("alignment_diagnostics"))

    blocked_reasons = _listish(pred.get("blocked_reasons") or pred.get("blocked_reasons_json"))
    degraded_reasons = _listish(pred.get("degraded_reasons") or pred.get("degraded_reasons_json"))

    return {
        "prediction_id": pred.get("prediction_id"),
        "prediction_business_key": pred.get("prediction_business_key"),
        "snapshot_id": pred.get("snapshot_id"),
        "ticker": pred.get("ticker"),
        "asof_ts_utc": pred.get("asof_ts_utc"),
        "horizon_kind": pred.get("horizon_kind"),
        "horizon_minutes": pred.get("horizon_minutes"),
        "horizon_seconds": pred.get("horizon_seconds"),
        "decision_window_id": pred.get("decision_window_id"),
        "decision_state": pred.get("decision_state"),
        "risk_gate_status": pred.get("risk_gate_status"),
        "data_quality_state": pred.get("data_quality_state"),
        "confidence_state": pred.get("confidence_state"),
        "suppression_reason": pred.get("suppression_reason") or probability_contract.get("suppression_reason") or meta_json.get("suppression_reason"),
        "ood_state": pred.get("ood_state") or probability_contract.get("ood_state") or meta_json.get("ood_state"),
        "replay_mode": pred.get("replay_mode") or meta_json.get("replay_mode"),
        "model_name": pred.get("model_name"),
        "model_version": pred.get("model_version"),
        "feature_version": pred.get("feature_version") or meta_json.get("feature_version"),
        "target_name": pred.get("target_name") or prediction_contract.get("target_name") or target_spec.get("target_name"),
        "target_version": pred.get("target_version") or prediction_contract.get("target_version") or target_spec.get("target_version"),
        "label_version": pred.get("label_version") or prediction_contract.get("label_version") or label_contract.get("label_version"),
        "calibration_version": pred.get("calibration_version") or calibration_ref.get("artifact_version"),
        "threshold_policy_version": pred.get("threshold_policy_version") or prediction_contract.get("threshold_policy_version") or label_contract.get("threshold_policy_version"),
        "endpoint_coverage": meta_json.get("endpoint_coverage"),
        "decision_dq": meta_json.get("decision_dq"),
        "alignment_status": pred.get("alignment_status"),
        "critical_missing_count": pred.get("critical_missing_count"),
        "blocked_reasons": blocked_reasons,
        "degraded_reasons": degraded_reasons,
        "dq_reason_codes": _listish(meta_json.get("dq_reason_codes")),
        "freshness_registry_diagnostics": freshness_diag,
        "alignment_diagnostics": alignment_diag,
        "probability_contract": probability_contract,
        "horizon_contract": horizon_contract,
    }


def classify_prediction_event(trace: Mapping[str, Any]) -> tuple[str, str, int]:
    suppression_reason = trace.get("suppression_reason")
    risk_gate_status = str(trace.get("risk_gate_status") or "").upper()
    data_quality_state = str(trace.get("data_quality_state") or "").upper()
    decision_state = str(trace.get("decision_state") or "").upper()

    if suppression_reason not in (None, "") or risk_gate_status == "BLOCKED" or decision_state == "NO_SIGNAL":
        return "signal_suppressed", "signal_suppressed_count", logging.WARNING
    if risk_gate_status == "DEGRADED" or data_quality_state in {"PARTIAL", "DEGRADED", "STALE"}:
        return "signal_degraded", "signal_degraded_count", logging.INFO
    return "signal_emitted", "signal_emitted_count", logging.INFO


def log_prediction_decision(
    logger: logging.Logger,
    prediction: Mapping[str, Any],
    *,
    ticker: Optional[str] = None,
    asof_ts_utc: Optional[Any] = None,
) -> Dict[str, Any]:
    trace = build_prediction_trace({**dict(prediction), "ticker": ticker, "asof_ts_utc": asof_ts_utc})
    suppression_reason = trace.get("suppression_reason")
    ood_state = str(trace.get("ood_state") or "").upper()

    if suppression_reason in {"MISSING_CALIBRATION_ARTIFACT", "INVALID_CALIBRATION_ARTIFACT", "CALIBRATION_TARGET_MISMATCH"}:
        structured_log(
            logger,
            logging.WARNING,
            event="calibration_missing" if suppression_reason == "MISSING_CALIBRATION_ARTIFACT" else "calibration_contract_invalid",
            msg="prediction probability contract calibration unavailable",
            counter="calibration_missing_count" if suppression_reason == "MISSING_CALIBRATION_ARTIFACT" else "calibration_contract_invalid_count",
            **trace,
        )

    if suppression_reason == "OOD_REJECTION" or ood_state == "OUT_OF_DISTRIBUTION":
        structured_log(
            logger,
            logging.WARNING,
            event="ood_rejection",
            msg="prediction rejected by ood policy",
            counter="ood_rejection_count",
            **trace,
        )

    event, counter, level = classify_prediction_event(trace)
    structured_log(
        logger,
        level,
        event=event,
        msg=f"prediction decision trace: {event}",
        counter=counter,
        **trace,
    )
    return trace
