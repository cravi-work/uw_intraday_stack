# src/storage.py
from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import math
import uuid
from collections import deque
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import duckdb

from .api_catalog_loader import EndpointRegistry
from .endpoint_truth import EndpointStateRow, to_utc_dt
from .logging_config import build_prediction_trace

logger = logging.getLogger(__name__)
UTC = timezone.utc

PUBLISH_TIME_KEYS: Tuple[str, ...] = (
    "source_publish_time",
    "source_publish_time_utc",
    "published_at",
    "publish_time",
    "report_time",
    "report_date",
    "updated_at",
    "last_updated",
)
SOURCE_REVISION_KEYS: Tuple[str, ...] = (
    "source_revision",
    "revision",
    "rev",
    "version",
    "sequence_id",
    "update_id",
)
FEATURE_POLICY_VERSION_FIELDS: Tuple[str, ...] = (
    "policy",
    "lag_class",
    "join_skew_tolerance_seconds",
    "max_tolerated_age_seconds",
    "policy_source",
)


class DbLockError(RuntimeError):
    pass


SCHEMA_SQL = """
CREATE SEQUENCE IF NOT EXISTS seq_endpoint_id START 1;
CREATE SEQUENCE IF NOT EXISTS seq_config_version START 1;

CREATE TABLE IF NOT EXISTS meta_runs (
    run_id UUID PRIMARY KEY,
    started_at_utc TIMESTAMP NOT NULL,
    ended_at_utc TIMESTAMP,
    asof_ts_utc TIMESTAMP NOT NULL,
    session_label TEXT,
    is_trading_day BOOLEAN,
    is_early_close BOOLEAN,
    config_version INTEGER,
    api_catalog_hash TEXT,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS snapshots (
    snapshot_id UUID PRIMARY KEY,
    run_id UUID REFERENCES meta_runs(run_id),
    asof_ts_utc TIMESTAMP NOT NULL,
    ticker TEXT NOT NULL,
    session_label TEXT,
    is_trading_day BOOLEAN,
    is_early_close BOOLEAN,
    data_quality_score DOUBLE,
    market_close_utc TIMESTAMP,
    post_end_utc TIMESTAMP,
    seconds_to_close INTEGER,
    created_at_utc TIMESTAMP DEFAULT current_timestamp,
    UNIQUE(ticker, asof_ts_utc)
);

CREATE TABLE IF NOT EXISTS meta_config (
    config_version INTEGER PRIMARY KEY,
    config_hash TEXT NOT NULL,
    config_yaml TEXT NOT NULL,
    created_at_utc TIMESTAMP DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS dim_endpoints (
    endpoint_id INTEGER PRIMARY KEY,
    method TEXT,
    path TEXT,
    signature TEXT UNIQUE,
    params_hash TEXT,
    params_json JSON
);

CREATE TABLE IF NOT EXISTS predictions (
    prediction_id UUID PRIMARY KEY,
    prediction_business_key TEXT NOT NULL,
    snapshot_id UUID REFERENCES snapshots(snapshot_id),
    horizon_minutes INTEGER,
    horizon_kind TEXT DEFAULT 'FIXED',
    horizon_seconds INTEGER,
    start_price DOUBLE,
    bias DOUBLE,
    confidence DOUBLE,
    prob_up DOUBLE,
    prob_down DOUBLE,
    prob_flat DOUBLE,
    target_name TEXT,
    target_version TEXT,
    label_version TEXT,
    feature_version TEXT,
    model_name TEXT,
    model_version TEXT,
    calibration_version TEXT,
    threshold_policy_version TEXT,
    replay_mode TEXT,
    ood_state TEXT,
    ood_reason TEXT,
    calibration_scope JSON,
    calibration_artifact_hash TEXT,
    decision_path_contract_version TEXT,
    suppression_reason TEXT,
    probability_contract_json JSON,
    model_hash TEXT,
    is_mock BOOLEAN DEFAULT FALSE,
    outcome_realized BOOLEAN DEFAULT FALSE,
    realized_at_utc TIMESTAMP,
    outcome_price DOUBLE,
    outcome_label TEXT,
    brier_score DOUBLE,
    log_loss DOUBLE,
    is_correct BOOLEAN,
    meta_json JSON,
    decision_state TEXT NOT NULL DEFAULT 'UNKNOWN',
    risk_gate_status TEXT NOT NULL DEFAULT 'UNKNOWN',
    data_quality_state TEXT NOT NULL DEFAULT 'UNKNOWN',
    confidence_state TEXT NOT NULL DEFAULT 'UNKNOWN',
    blocked_reasons_json JSON,
    degraded_reasons_json JSON,
    validation_eligible BOOLEAN NOT NULL DEFAULT TRUE,
    gate_json JSON,
    alignment_status TEXT NOT NULL DEFAULT 'UNKNOWN',
    source_ts_min_utc TIMESTAMP NULL,
    source_ts_max_utc TIMESTAMP NULL,
    critical_missing_count INTEGER NOT NULL DEFAULT 0,
    decision_window_id TEXT NOT NULL DEFAULT 'UNKNOWN'
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_preds_dedupe ON predictions (snapshot_id, horizon_kind, horizon_minutes, horizon_seconds);

CREATE TABLE IF NOT EXISTS features (
    snapshot_id UUID REFERENCES snapshots(snapshot_id),
    feature_key TEXT,
    feature_value DOUBLE,
    meta_json JSON,
    UNIQUE(snapshot_id, feature_key)
);

CREATE TABLE IF NOT EXISTS derived_levels (
    snapshot_id UUID,
    level_type TEXT,
    price DOUBLE,
    magnitude DOUBLE,
    meta_json JSON
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_derived_levels_dedupe ON derived_levels (snapshot_id, level_type, price);

CREATE TABLE IF NOT EXISTS snapshot_lineage (
    snapshot_id UUID NOT NULL REFERENCES snapshots(snapshot_id),
    endpoint_id INTEGER,
    used_event_id UUID,
    freshness_state TEXT,
    data_age_seconds INTEGER,
    payload_class TEXT,
    na_reason TEXT,
    meta_json JSON,
    UNIQUE(snapshot_id, endpoint_id)
);

CREATE TABLE IF NOT EXISTS endpoint_state (
    ticker TEXT,
    endpoint_id INTEGER,
    last_success_event_id UUID,
    last_success_ts_utc TIMESTAMP,
    last_payload_hash TEXT,
    last_change_ts_utc TIMESTAMP,
    last_change_event_id UUID,
    last_attempt_event_id UUID,
    last_attempt_ts_utc TIMESTAMP,
    last_attempt_http_status INTEGER,
    last_attempt_error_type TEXT,
    last_attempt_error_msg TEXT,
    PRIMARY KEY (ticker, endpoint_id)
);

CREATE TABLE IF NOT EXISTS raw_http_events (
    event_id UUID PRIMARY KEY,
    run_id UUID,
    requested_at_utc TIMESTAMP,
    received_at_utc TIMESTAMP,
    ticker TEXT,
    endpoint_id INTEGER,
    http_status INTEGER,
    latency_ms INTEGER,
    payload_hash TEXT,
    payload_json JSON,
    source_publish_time_utc TIMESTAMP,
    source_revision TEXT,
    is_retry BOOLEAN,
    error_type TEXT,
    error_msg TEXT,
    circuit_state_json JSON
);

CREATE TABLE IF NOT EXISTS decision_traces (
    trace_id UUID PRIMARY KEY,
    created_at_utc TIMESTAMP NOT NULL,
    prediction_id UUID,
    prediction_business_key TEXT NOT NULL,
    snapshot_id UUID REFERENCES snapshots(snapshot_id),
    event_type TEXT NOT NULL,
    decision_state TEXT,
    risk_gate_status TEXT,
    data_quality_state TEXT,
    confidence_state TEXT,
    suppression_reason TEXT,
    ood_state TEXT,
    ood_reason TEXT,
    replay_mode TEXT,
    model_name TEXT,
    model_version TEXT,
    target_name TEXT,
    target_version TEXT,
    calibration_version TEXT,
    calibration_scope JSON,
    calibration_artifact_hash TEXT,
    decision_path_contract_version TEXT,
    threshold_policy_version TEXT,
    blocked_reasons_json JSON,
    degraded_reasons_json JSON,
    trace_json JSON
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_decision_trace_dedupe ON decision_traces (prediction_business_key, event_type);

CREATE TABLE IF NOT EXISTS config_history (
    config_version VARCHAR,
    ingested_at_utc TIMESTAMP,
    yaml_content VARCHAR
);

CREATE TABLE IF NOT EXISTS dim_tickers (ticker TEXT PRIMARY KEY);

CREATE INDEX IF NOT EXISTS idx_snapshots_ticker_asof ON snapshots(ticker, asof_ts_utc);
CREATE INDEX IF NOT EXISTS idx_features_snap ON features(snapshot_id, feature_key);
CREATE INDEX IF NOT EXISTS idx_lineage_snap ON snapshot_lineage(snapshot_id);
"""

REPLAY_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS replay_runs (
    replay_run_id UUID PRIMARY KEY,
    started_at_utc TIMESTAMP NOT NULL,
    finished_at_utc TIMESTAMP,
    ticker TEXT NOT NULL,
    replay_mode TEXT NOT NULL,
    requested_start_ts_utc TIMESTAMP,
    requested_end_ts_utc TIMESTAMP,
    snapshot_count INTEGER NOT NULL DEFAULT 0,
    prediction_count INTEGER NOT NULL DEFAULT 0,
    model_name TEXT,
    model_version TEXT,
    feature_version TEXT,
    calibration_version TEXT,
    threshold_policy_version TEXT,
    target_name TEXT,
    target_version TEXT,
    label_version TEXT,
    contract_json JSON,
    status TEXT NOT NULL,
    failure_reason TEXT
);

CREATE INDEX IF NOT EXISTS idx_replay_runs_ticker_started ON replay_runs(ticker, started_at_utc);
"""


def _safe_json_loads(raw: Any) -> Dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except Exception:
            return {}
        return dict(parsed) if isinstance(parsed, dict) else {}
    return {}


def _safe_json_dumps(raw: Any, default: Any = None) -> Optional[str]:
    if raw is None:
        raw = default
    if raw is None:
        return None
    return json.dumps(raw)


def _coerce_optional_utc_dt(x: Any) -> Optional[datetime]:
    if x is None:
        return None
    if isinstance(x, datetime):
        if x.tzinfo is None:
            return x.replace(tzinfo=UTC)
        return x.astimezone(UTC)
    if isinstance(x, (int, float)):
        if not math.isfinite(float(x)):
            return None
        try:
            return datetime.fromtimestamp(float(x), UTC)
        except (OverflowError, OSError, ValueError):
            return None
    if isinstance(x, str):
        raw = x.strip()
        if not raw:
            return None
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(UTC)
        except ValueError:
            try:
                return datetime.fromtimestamp(float(raw), UTC)
            except (TypeError, ValueError, OverflowError, OSError):
                return None
    return None


def _find_first_nested_value(payload: Any, candidate_keys: Sequence[str], *, max_nodes: int = 256) -> Any:
    if payload is None:
        return None

    keyset = {str(k) for k in candidate_keys}
    queue: deque[Any] = deque([payload])
    seen: set[int] = set()
    visited = 0

    while queue and visited < max_nodes:
        current = queue.popleft()
        visited += 1
        current_id = id(current)
        if current_id in seen:
            continue
        seen.add(current_id)

        if isinstance(current, Mapping):
            for key in keyset:
                if key in current:
                    value = current.get(key)
                    if value not in (None, ""):
                        return value
            for value in current.values():
                if isinstance(value, (Mapping, list, tuple)):
                    queue.append(value)
        elif isinstance(current, (list, tuple)):
            for value in current[:32]:
                if isinstance(value, (Mapping, list, tuple)):
                    queue.append(value)
    return None


def _infer_source_lineage_from_payload(payload: Any) -> Tuple[Optional[datetime], Optional[str]]:
    publish_raw = _find_first_nested_value(payload, PUBLISH_TIME_KEYS)
    revision_raw = _find_first_nested_value(payload, SOURCE_REVISION_KEYS)
    return _coerce_optional_utc_dt(publish_raw), str(revision_raw) if revision_raw not in (None, "") else None


def _normalize_horizon_fields(
    *,
    horizon_kind: Any,
    horizon_minutes: Any,
    horizon_seconds: Any,
) -> Tuple[str, Optional[int], Optional[int]]:
    kind = str(horizon_kind or "FIXED").upper()
    minutes: Optional[int]
    seconds: Optional[int]

    minutes = None if horizon_minutes is None else int(horizon_minutes)
    seconds = None if horizon_seconds is None else int(horizon_seconds)

    if kind == "TO_CLOSE":
        if minutes is None:
            minutes = 0
        if seconds is None:
            raise ValueError("TO_CLOSE prediction requires horizon_seconds")
    return kind, minutes, seconds


def _prediction_business_key(snapshot_id: str, horizon_kind: str, horizon_minutes: Optional[int], horizon_seconds: Optional[int]) -> str:
    return json.dumps(
        {
            "snapshot_id": str(snapshot_id),
            "horizon_kind": str(horizon_kind),
            "horizon_minutes": horizon_minutes,
            "horizon_seconds": horizon_seconds,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _prediction_id_from_business_key(business_key: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"uw_intraday_stack/prediction/{business_key}"))


def _decision_trace_id(prediction_business_key: str, event_type: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"uw_intraday_stack/decision_trace/{prediction_business_key}/{event_type}"))


def _derive_feature_version(meta_json: Dict[str, Any]) -> Optional[str]:
    explicit = meta_json.get("feature_version") or meta_json.get("feature_contract_version")
    if explicit not in (None, ""):
        return str(explicit)

    freshness = meta_json.get("freshness_registry_diagnostics")
    if not isinstance(freshness, dict):
        return None

    policies = freshness.get("feature_policies")
    if not isinstance(policies, dict) or not policies:
        return None

    contract_view = {
        str(key): {
            field: (value.get(field) if isinstance(value, dict) else None)
            for field in FEATURE_POLICY_VERSION_FIELDS
        }
        for key, value in policies.items()
    }
    digest = hashlib.sha256(json.dumps(contract_view, sort_keys=True).encode()).hexdigest()[:16]
    return f"derived_feature_contract_{digest}"


def _as_mapping(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, Mapping):
        return dict(raw)
    if isinstance(raw, str):
        return _safe_json_loads(raw)
    return {}


def _normalize_jsonish_mapping(raw: Any) -> Optional[Dict[str, Any]]:
    mapping = _as_mapping(raw)
    return mapping or None


def _infer_decision_path_contract_version(meta_json: Dict[str, Any], prediction_contract: Dict[str, Any]) -> Optional[str]:
    explicit = meta_json.get("decision_path_contract_version") or prediction_contract.get("decision_path_contract_version")
    if explicit not in (None, ""):
        return str(explicit)

    horizon_contract = _as_mapping(meta_json.get("horizon_contract"))
    explicit = horizon_contract.get("decision_path_contract_version")
    if explicit not in (None, ""):
        return str(explicit)

    feature_contracts = horizon_contract.get("feature_contracts")
    if isinstance(feature_contracts, Mapping):
        versions = sorted({
            str(
                contract.get("contract_version")
                or contract.get("feature_use_contract_version")
                or contract.get("purpose_contract_version")
            ).strip()
            for contract in feature_contracts.values()
            if isinstance(contract, Mapping)
            and (
                contract.get("contract_version")
                or contract.get("feature_use_contract_version")
                or contract.get("purpose_contract_version")
            ) not in (None, "")
        })
        if len(versions) == 1:
            return versions[0]
        if len(versions) > 1:
            return f"MIXED:{','.join(versions)}"
    return None


def _extract_prediction_contract_fields(prediction: Mapping[str, Any]) -> Dict[str, Any]:
    meta_json = _safe_json_loads(prediction.get("meta_json"))
    probability_contract = _as_mapping(meta_json.get("probability_contract"))
    prediction_contract = _as_mapping(meta_json.get("prediction_contract"))

    target_spec = _as_mapping(prediction_contract.get("target_spec"))
    if not target_spec:
        target_spec = _as_mapping(probability_contract.get("target_spec"))

    label_contract = _as_mapping(prediction_contract.get("label_contract"))
    calibration_ref = _as_mapping(probability_contract.get("calibration_artifact_ref"))
    calibration_selection = _as_mapping(meta_json.get("calibration_selection"))
    ood_assessment = _as_mapping(meta_json.get("ood_assessment"))

    target_name = prediction.get("target_name") or prediction_contract.get("target_name") or target_spec.get("target_name")
    target_version = prediction.get("target_version") or prediction_contract.get("target_version") or target_spec.get("target_version")
    label_version = prediction.get("label_version") or prediction_contract.get("label_version") or label_contract.get("label_version")
    threshold_policy_version = (
        prediction.get("threshold_policy_version")
        or prediction_contract.get("threshold_policy_version")
        or label_contract.get("threshold_policy_version")
    )
    calibration_version = prediction.get("calibration_version") or calibration_ref.get("artifact_version")
    feature_version = prediction.get("feature_version") or _derive_feature_version(meta_json)
    replay_mode = prediction.get("replay_mode") or meta_json.get("replay_mode")
    ood_state = prediction.get("ood_state") or meta_json.get("ood_state") or probability_contract.get("ood_state")
    ood_reason = (
        prediction.get("ood_reason")
        or meta_json.get("ood_reason")
        or ood_assessment.get("primary_reason")
        or probability_contract.get("ood_reason")
    )
    calibration_scope = (
        _normalize_jsonish_mapping(prediction.get("calibration_scope"))
        or _normalize_jsonish_mapping(prediction_contract.get("calibration_scope"))
        or _normalize_jsonish_mapping(calibration_selection.get("calibration_scope"))
        or _normalize_jsonish_mapping(calibration_ref.get("calibration_scope"))
    )
    calibration_artifact_hash = (
        prediction.get("calibration_artifact_hash")
        or calibration_selection.get("artifact_hash")
        or calibration_ref.get("artifact_hash")
    )
    decision_path_contract_version = (
        prediction.get("decision_path_contract_version")
        or _infer_decision_path_contract_version(meta_json, prediction_contract)
    )
    suppression_reason = (
        prediction.get("suppression_reason")
        or meta_json.get("suppression_reason")
        or probability_contract.get("suppression_reason")
    )

    return {
        "target_name": str(target_name) if target_name not in (None, "") else None,
        "target_version": str(target_version) if target_version not in (None, "") else None,
        "label_version": str(label_version) if label_version not in (None, "") else None,
        "feature_version": str(feature_version) if feature_version not in (None, "") else None,
        "calibration_version": str(calibration_version) if calibration_version not in (None, "") else None,
        "threshold_policy_version": str(threshold_policy_version) if threshold_policy_version not in (None, "") else None,
        "replay_mode": str(replay_mode) if replay_mode not in (None, "") else None,
        "ood_state": str(ood_state) if ood_state not in (None, "") else None,
        "ood_reason": str(ood_reason) if ood_reason not in (None, "") else None,
        "calibration_scope": calibration_scope if calibration_scope else None,
        "calibration_artifact_hash": str(calibration_artifact_hash) if calibration_artifact_hash not in (None, "") else None,
        "decision_path_contract_version": str(decision_path_contract_version) if decision_path_contract_version not in (None, "") else None,
        "suppression_reason": str(suppression_reason) if suppression_reason not in (None, "") else None,
        "probability_contract_json": probability_contract if probability_contract else None,
    }


def extract_prediction_contract_fields(prediction: Mapping[str, Any]) -> Dict[str, Any]:
    return _extract_prediction_contract_fields(prediction)



class DbWriter:
    def __init__(self, duckdb_path: str, lock_path: str = "uw.lock"):
        self.duckdb_path = duckdb_path
        self.lock_path = lock_path

    def _connect_new(self) -> duckdb.DuckDBPyConnection:
        con = duckdb.connect(self.duckdb_path)
        con.execute("PRAGMA threads=4")
        return con

    def ensure_schema(self, con: duckdb.DuckDBPyConnection) -> None:
        con.execute(SCHEMA_SQL)
        self.ensure_replay_schema(con)
        self._migrate_additive(con)

    def ensure_replay_schema(self, con: duckdb.DuckDBPyConnection) -> None:
        con.execute(REPLAY_SCHEMA_SQL)

    def _migrate_additive(self, con: duckdb.DuckDBPyConnection) -> None:
        def _add(table: str, column: str, typ: str) -> None:
            cols = [r[1] for r in con.execute(f"PRAGMA table_info('{table}')").fetchall()]
            if column not in cols:
                con.execute(f"ALTER TABLE {table} ADD COLUMN {column} {typ}")

        _add("snapshots", "is_early_close", "BOOLEAN")
        _add("snapshots", "market_close_utc", "TIMESTAMP")
        _add("snapshots", "post_end_utc", "TIMESTAMP")
        _add("snapshots", "seconds_to_close", "INTEGER")

        _add("predictions", "prediction_business_key", "TEXT")
        _add("predictions", "prob_flat", "DOUBLE")
        _add("predictions", "horizon_kind", "TEXT DEFAULT 'FIXED'")
        _add("predictions", "horizon_seconds", "INTEGER")
        _add("predictions", "target_name", "TEXT")
        _add("predictions", "target_version", "TEXT")
        _add("predictions", "label_version", "TEXT")
        _add("predictions", "feature_version", "TEXT")
        _add("predictions", "calibration_version", "TEXT")
        _add("predictions", "threshold_policy_version", "TEXT")
        _add("predictions", "replay_mode", "TEXT")
        _add("predictions", "ood_state", "TEXT")
        _add("predictions", "ood_reason", "TEXT")
        _add("predictions", "calibration_scope", "JSON")
        _add("predictions", "calibration_artifact_hash", "TEXT")
        _add("predictions", "decision_path_contract_version", "TEXT")
        _add("predictions", "suppression_reason", "TEXT")
        _add("predictions", "probability_contract_json", "JSON")
        _add("predictions", "is_mock", "BOOLEAN DEFAULT FALSE")
        _add("predictions", "outcome_price", "DOUBLE")
        _add("predictions", "is_correct", "BOOLEAN")
        _add("predictions", "meta_json", "JSON")
        _add("predictions", "decision_state", "TEXT DEFAULT 'UNKNOWN'")
        _add("predictions", "risk_gate_status", "TEXT DEFAULT 'UNKNOWN'")
        _add("predictions", "data_quality_state", "TEXT DEFAULT 'UNKNOWN'")
        _add("predictions", "confidence_state", "TEXT DEFAULT 'UNKNOWN'")
        _add("predictions", "blocked_reasons_json", "JSON")
        _add("predictions", "degraded_reasons_json", "JSON")
        _add("predictions", "validation_eligible", "BOOLEAN DEFAULT TRUE")
        _add("predictions", "gate_json", "JSON")
        _add("predictions", "alignment_status", "TEXT DEFAULT 'UNKNOWN'")
        _add("predictions", "source_ts_min_utc", "TIMESTAMP NULL")
        _add("predictions", "source_ts_max_utc", "TIMESTAMP NULL")
        _add("predictions", "critical_missing_count", "INTEGER DEFAULT 0")
        _add("predictions", "decision_window_id", "TEXT DEFAULT 'UNKNOWN'")

        _add("endpoint_state", "last_change_ts_utc", "TIMESTAMP")
        _add("endpoint_state", "last_change_event_id", "UUID")
        _add("endpoint_state", "last_attempt_event_id", "UUID")
        _add("endpoint_state", "last_attempt_ts_utc", "TIMESTAMP")
        _add("endpoint_state", "last_attempt_http_status", "INTEGER")
        _add("endpoint_state", "last_attempt_error_type", "TEXT")
        _add("endpoint_state", "last_attempt_error_msg", "TEXT")

        _add("snapshot_lineage", "payload_class", "TEXT")
        _add("snapshot_lineage", "na_reason", "TEXT")
        _add("snapshot_lineage", "meta_json", "JSON")

        _add("raw_http_events", "source_publish_time_utc", "TIMESTAMP")
        _add("raw_http_events", "source_revision", "TEXT")

        con.execute(
            """
            CREATE TABLE IF NOT EXISTS decision_traces (
                trace_id UUID PRIMARY KEY,
                created_at_utc TIMESTAMP NOT NULL,
                prediction_id UUID,
                prediction_business_key TEXT NOT NULL,
                snapshot_id UUID REFERENCES snapshots(snapshot_id),
                event_type TEXT NOT NULL,
                decision_state TEXT,
                risk_gate_status TEXT,
                data_quality_state TEXT,
                confidence_state TEXT,
                suppression_reason TEXT,
                ood_state TEXT,
                ood_reason TEXT,
                replay_mode TEXT,
                model_name TEXT,
                model_version TEXT,
                target_name TEXT,
                target_version TEXT,
                calibration_version TEXT,
                calibration_scope JSON,
                calibration_artifact_hash TEXT,
                decision_path_contract_version TEXT,
                threshold_policy_version TEXT,
                blocked_reasons_json JSON,
                degraded_reasons_json JSON,
                trace_json JSON
            )
            """
        )
        con.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_decision_trace_dedupe ON decision_traces (prediction_business_key, event_type)")
        _add("decision_traces", "ood_reason", "TEXT")
        _add("decision_traces", "calibration_scope", "JSON")
        _add("decision_traces", "calibration_artifact_hash", "TEXT")
        _add("decision_traces", "decision_path_contract_version", "TEXT")

        pred_cols = {r[1]: r[2] for r in con.execute("PRAGMA table_info('predictions')").fetchall()}
        if pred_cols.get("bias") in ["VARCHAR", "TEXT"]:
            logger.info("Migrating predictions.bias from TEXT to DOUBLE")
            con.execute("ALTER TABLE predictions ALTER bias TYPE DOUBLE USING TRY_CAST(bias AS DOUBLE)")
            con.execute(
                """
                UPDATE predictions
                SET meta_json = json_insert(COALESCE(meta_json, '{}'), '$.migration_note', 'bias_cast_failed')
                WHERE bias IS NULL AND meta_json NOT LIKE '%bias_cast_failed%'
                """
            )

        con.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_predictions_business_key ON predictions(prediction_business_key)")
        self._backfill_prediction_contract_columns(con)
        self._backfill_decision_trace_governance_columns(con)
        self._backfill_raw_http_event_lineage(con)

    def _backfill_prediction_contract_columns(self, con: duckdb.DuckDBPyConnection) -> None:
        rows = con.execute(
            """
            SELECT prediction_id, snapshot_id, horizon_kind, horizon_minutes, horizon_seconds,
                   prediction_business_key, meta_json, target_name, target_version, label_version,
                   feature_version, calibration_version, threshold_policy_version, replay_mode,
                   ood_state, ood_reason, calibration_scope, calibration_artifact_hash,
                   decision_path_contract_version, suppression_reason, probability_contract_json
            FROM predictions
            """
        ).fetchall()

        updated = 0
        for row in rows:
            (
                prediction_id,
                snapshot_id,
                horizon_kind,
                horizon_minutes,
                horizon_seconds,
                business_key,
                meta_json_raw,
                target_name,
                target_version,
                label_version,
                feature_version,
                calibration_version,
                threshold_policy_version,
                replay_mode,
                ood_state,
                ood_reason,
                calibration_scope,
                calibration_artifact_hash,
                decision_path_contract_version,
                suppression_reason,
                probability_contract_json,
            ) = row

            kind, minutes, seconds = _normalize_horizon_fields(
                horizon_kind=horizon_kind,
                horizon_minutes=horizon_minutes,
                horizon_seconds=horizon_seconds,
            )
            stable_key = business_key or _prediction_business_key(str(snapshot_id), kind, minutes, seconds)
            extracted = _extract_prediction_contract_fields({"meta_json": meta_json_raw})

            changed = (
                business_key != stable_key
                or (target_name in (None, "") and extracted["target_name"] is not None)
                or (target_version in (None, "") and extracted["target_version"] is not None)
                or (label_version in (None, "") and extracted["label_version"] is not None)
                or (feature_version in (None, "") and extracted["feature_version"] is not None)
                or (calibration_version in (None, "") and extracted["calibration_version"] is not None)
                or (threshold_policy_version in (None, "") and extracted["threshold_policy_version"] is not None)
                or (replay_mode in (None, "") and extracted["replay_mode"] is not None)
                or (ood_state in (None, "") and extracted["ood_state"] is not None)
                or (ood_reason in (None, "") and extracted["ood_reason"] is not None)
                or (_normalize_jsonish_mapping(calibration_scope) is None and extracted["calibration_scope"] is not None)
                or (calibration_artifact_hash in (None, "") and extracted["calibration_artifact_hash"] is not None)
                or (decision_path_contract_version in (None, "") and extracted["decision_path_contract_version"] is not None)
                or (suppression_reason in (None, "") and extracted["suppression_reason"] is not None)
                or (probability_contract_json is None and extracted["probability_contract_json"] is not None)
            )
            if not changed:
                continue

            con.execute(
                """
                UPDATE predictions
                SET prediction_business_key = ?,
                    target_name = COALESCE(target_name, ?),
                    target_version = COALESCE(target_version, ?),
                    label_version = COALESCE(label_version, ?),
                    feature_version = COALESCE(feature_version, ?),
                    calibration_version = COALESCE(calibration_version, ?),
                    threshold_policy_version = COALESCE(threshold_policy_version, ?),
                    replay_mode = COALESCE(replay_mode, ?),
                    ood_state = COALESCE(ood_state, ?),
                    ood_reason = COALESCE(ood_reason, ?),
                    calibration_scope = COALESCE(calibration_scope, ?),
                    calibration_artifact_hash = COALESCE(calibration_artifact_hash, ?),
                    decision_path_contract_version = COALESCE(decision_path_contract_version, ?),
                    suppression_reason = COALESCE(suppression_reason, ?),
                    probability_contract_json = COALESCE(probability_contract_json, ?)
                WHERE prediction_id = ?
                """,
                [
                    stable_key,
                    extracted["target_name"],
                    extracted["target_version"],
                    extracted["label_version"],
                    extracted["feature_version"],
                    extracted["calibration_version"],
                    extracted["threshold_policy_version"],
                    extracted["replay_mode"],
                    extracted["ood_state"],
                    extracted["ood_reason"],
                    _safe_json_dumps(extracted["calibration_scope"]),
                    extracted["calibration_artifact_hash"],
                    extracted["decision_path_contract_version"],
                    extracted["suppression_reason"],
                    _safe_json_dumps(extracted["probability_contract_json"]),
                    prediction_id,
                ],
            )
            updated += 1

        if updated:
            logger.info("Backfilled prediction contract columns", extra={"json": {"rows": updated}})

    def _backfill_decision_trace_governance_columns(self, con: duckdb.DuckDBPyConnection) -> None:
        pred_lookup = {
            str(row[0]): {
                "ood_reason": row[1],
                "calibration_scope": _normalize_jsonish_mapping(row[2]),
                "calibration_artifact_hash": row[3],
                "decision_path_contract_version": row[4],
            }
            for row in con.execute(
                """
                SELECT prediction_business_key, ood_reason, calibration_scope,
                       calibration_artifact_hash, decision_path_contract_version
                FROM predictions
                WHERE prediction_business_key IS NOT NULL
                """
            ).fetchall()
        }

        rows = con.execute(
            """
            SELECT trace_id, prediction_business_key, trace_json, ood_reason, calibration_scope,
                   calibration_artifact_hash, decision_path_contract_version
            FROM decision_traces
            """
        ).fetchall()

        updated = 0
        for trace_id, business_key, trace_json_raw, current_ood_reason, current_scope_raw, current_artifact_hash, current_dp_version in rows:
            trace_json = _safe_json_loads(trace_json_raw)
            prediction_values = pred_lookup.get(str(business_key), {})

            extracted_ood_reason = current_ood_reason or trace_json.get("ood_reason") or prediction_values.get("ood_reason")
            extracted_scope = (
                _normalize_jsonish_mapping(current_scope_raw)
                or _normalize_jsonish_mapping(trace_json.get("calibration_scope"))
                or prediction_values.get("calibration_scope")
            )
            extracted_artifact_hash = current_artifact_hash or trace_json.get("calibration_artifact_hash") or prediction_values.get("calibration_artifact_hash")
            extracted_dp_version = current_dp_version or trace_json.get("decision_path_contract_version") or prediction_values.get("decision_path_contract_version")

            changed = (
                (current_ood_reason in (None, "") and extracted_ood_reason is not None)
                or (_normalize_jsonish_mapping(current_scope_raw) is None and extracted_scope is not None)
                or (current_artifact_hash in (None, "") and extracted_artifact_hash is not None)
                or (current_dp_version in (None, "") and extracted_dp_version is not None)
            )
            if not changed:
                continue

            con.execute(
                """
                UPDATE decision_traces
                SET ood_reason = COALESCE(ood_reason, ?),
                    calibration_scope = COALESCE(calibration_scope, ?),
                    calibration_artifact_hash = COALESCE(calibration_artifact_hash, ?),
                    decision_path_contract_version = COALESCE(decision_path_contract_version, ?)
                WHERE trace_id = ?
                """,
                [
                    extracted_ood_reason,
                    _safe_json_dumps(extracted_scope),
                    extracted_artifact_hash,
                    extracted_dp_version,
                    trace_id,
                ],
            )
            updated += 1

        if updated:
            logger.info("Backfilled decision trace governance columns", extra={"json": {"rows": updated}})

    def _backfill_raw_http_event_lineage(self, con: duckdb.DuckDBPyConnection) -> None:
        rows = con.execute(
            """
            SELECT event_id, payload_json, source_publish_time_utc, source_revision
            FROM raw_http_events
            WHERE source_publish_time_utc IS NULL OR source_revision IS NULL
            """
        ).fetchall()

        updated = 0
        for event_id, payload_json_raw, source_publish_time_utc, source_revision in rows:
            payload = payload_json_raw
            if isinstance(payload_json_raw, str):
                try:
                    payload = json.loads(payload_json_raw)
                except Exception:
                    payload = None
            inferred_publish, inferred_revision = _infer_source_lineage_from_payload(payload)
            if inferred_publish is None and inferred_revision is None:
                continue

            con.execute(
                """
                UPDATE raw_http_events
                SET source_publish_time_utc = COALESCE(source_publish_time_utc, ?),
                    source_revision = COALESCE(source_revision, ?)
                WHERE event_id = ?
                """,
                [inferred_publish, inferred_revision, event_id],
            )
            updated += 1

        if updated:
            logger.info("Backfilled raw event lineage columns", extra={"json": {"rows": updated}})

    def begin_replay_run(
        self,
        con: duckdb.DuckDBPyConnection,
        *,
        ticker: str,
        replay_mode: str,
        started_at_utc: datetime,
        requested_start_ts_utc: Optional[datetime] = None,
        requested_end_ts_utc: Optional[datetime] = None,
        status: str = "RUNNING",
        contract: Optional[Mapping[str, Any]] = None,
    ) -> str:
        self.ensure_replay_schema(con)
        replay_run_id = str(uuid.uuid4())
        contract_dict = dict(contract or {})
        con.execute(
            """
            INSERT INTO replay_runs (
                replay_run_id, started_at_utc, ticker, replay_mode,
                requested_start_ts_utc, requested_end_ts_utc,
                model_name, model_version, feature_version, calibration_version,
                threshold_policy_version, target_name, target_version, label_version,
                contract_json, status, failure_reason
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            [
                replay_run_id,
                started_at_utc,
                ticker.upper(),
                replay_mode,
                requested_start_ts_utc,
                requested_end_ts_utc,
                contract_dict.get("model_name"),
                contract_dict.get("model_version"),
                contract_dict.get("feature_version"),
                contract_dict.get("calibration_version"),
                contract_dict.get("threshold_policy_version"),
                contract_dict.get("target_name"),
                contract_dict.get("target_version"),
                contract_dict.get("label_version"),
                _safe_json_dumps(contract_dict, default={}),
                status,
                None,
            ],
        )
        return replay_run_id

    def finish_replay_run(
        self,
        con: duckdb.DuckDBPyConnection,
        replay_run_id: str,
        *,
        finished_at_utc: datetime,
        status: str,
        snapshot_count: int,
        prediction_count: int,
        failure_reason: Optional[str] = None,
        contract: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.ensure_replay_schema(con)
        contract_dict = dict(contract or {})
        con.execute(
            """
            UPDATE replay_runs
            SET finished_at_utc = ?,
                snapshot_count = ?,
                prediction_count = ?,
                model_name = COALESCE(?, model_name),
                model_version = COALESCE(?, model_version),
                feature_version = COALESCE(?, feature_version),
                calibration_version = COALESCE(?, calibration_version),
                threshold_policy_version = COALESCE(?, threshold_policy_version),
                target_name = COALESCE(?, target_name),
                target_version = COALESCE(?, target_version),
                label_version = COALESCE(?, label_version),
                contract_json = COALESCE(?, contract_json),
                status = ?,
                failure_reason = ?
            WHERE replay_run_id = ?
            """,
            [
                finished_at_utc,
                int(snapshot_count),
                int(prediction_count),
                contract_dict.get("model_name"),
                contract_dict.get("model_version"),
                contract_dict.get("feature_version"),
                contract_dict.get("calibration_version"),
                contract_dict.get("threshold_policy_version"),
                contract_dict.get("target_name"),
                contract_dict.get("target_version"),
                contract_dict.get("label_version"),
                _safe_json_dumps(contract_dict, default={}) if contract_dict else None,
                status,
                failure_reason[:512] if failure_reason else None,
                replay_run_id,
            ],
        )

    @contextlib.contextmanager
    def writer(self):
        con = self._connect_new()
        con.execute("BEGIN TRANSACTION")
        try:
            yield con
            con.execute("COMMIT")
        except Exception as exc:
            con.execute("ROLLBACK")
            logger.error("Transaction rollback triggered", extra={"json": {"error": str(exc)}})
            raise
        finally:
            con.close()

    def get_payloads_by_event_ids(self, con: duckdb.DuckDBPyConnection, event_ids: List[str]) -> Dict[str, Any]:
        if not event_ids:
            return {}
        placeholders = ",".join(["?"] * len(event_ids))
        rows = con.execute(
            f"SELECT event_id, payload_json FROM raw_http_events WHERE event_id IN ({placeholders})",
            event_ids,
        ).fetchall()

        out: Dict[str, Any] = {}
        for event_id, payload_json in rows:
            key = str(event_id)
            if payload_json is None:
                out[key] = None
                continue
            try:
                out[key] = json.loads(payload_json) if isinstance(payload_json, str) else payload_json
            except Exception:
                out[key] = None
        return out

    def insert_snapshot(
        self,
        con: duckdb.DuckDBPyConnection,
        *,
        run_id,
        asof_ts_utc,
        ticker,
        session_label,
        is_trading_day,
        is_early_close: bool,
        data_quality_score,
        market_close_utc: Optional[datetime],
        post_end_utc: Optional[datetime],
        seconds_to_close: Optional[int],
    ) -> str:
        row = con.execute(
            "SELECT snapshot_id FROM snapshots WHERE ticker=? AND asof_ts_utc=?",
            [ticker.upper(), asof_ts_utc],
        ).fetchone()
        if row:
            return str(row[0])

        snapshot_id = uuid.uuid4()
        con.execute(
            """
            INSERT INTO snapshots (
                snapshot_id, run_id, asof_ts_utc, ticker, session_label, is_trading_day,
                is_early_close, data_quality_score, market_close_utc, post_end_utc,
                seconds_to_close, created_at_utc
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            [
                str(snapshot_id),
                str(run_id) if run_id is not None else None,
                asof_ts_utc,
                ticker.upper(),
                session_label,
                is_trading_day,
                is_early_close,
                data_quality_score,
                market_close_utc,
                post_end_utc,
                seconds_to_close,
                datetime.now(UTC),
            ],
        )
        return str(snapshot_id)

    def insert_prediction(self, con: duckdb.DuckDBPyConnection, p: Mapping[str, Any]) -> str:
        snapshot_id = str(p["snapshot_id"])
        horizon_kind, horizon_minutes, horizon_seconds = _normalize_horizon_fields(
            horizon_kind=p.get("horizon_kind", "FIXED"),
            horizon_minutes=p.get("horizon_minutes"),
            horizon_seconds=p.get("horizon_seconds"),
        )

        meta_json = _safe_json_loads(p.get("meta_json"))
        extracted = _extract_prediction_contract_fields({**dict(p), "meta_json": meta_json})
        business_key = str(
            p.get("prediction_business_key")
            or _prediction_business_key(snapshot_id, horizon_kind, horizon_minutes, horizon_seconds)
        )
        prediction_id = str(p.get("prediction_id") or _prediction_id_from_business_key(business_key))

        row = {
            "prediction_id": prediction_id,
            "prediction_business_key": business_key,
            "snapshot_id": snapshot_id,
            "horizon_minutes": horizon_minutes,
            "horizon_kind": horizon_kind,
            "horizon_seconds": horizon_seconds,
            "start_price": p.get("start_price"),
            "bias": p.get("bias"),
            "confidence": p.get("confidence"),
            "prob_up": p.get("prob_up"),
            "prob_down": p.get("prob_down"),
            "prob_flat": p.get("prob_flat"),
            "target_name": extracted["target_name"],
            "target_version": extracted["target_version"],
            "label_version": extracted["label_version"],
            "feature_version": extracted["feature_version"],
            "model_name": p.get("model_name"),
            "model_version": p.get("model_version"),
            "calibration_version": extracted["calibration_version"],
            "threshold_policy_version": extracted["threshold_policy_version"],
            "replay_mode": extracted["replay_mode"],
            "ood_state": extracted["ood_state"],
            "ood_reason": extracted["ood_reason"],
            "calibration_scope": _safe_json_dumps(extracted["calibration_scope"]),
            "calibration_artifact_hash": extracted["calibration_artifact_hash"],
            "decision_path_contract_version": extracted["decision_path_contract_version"],
            "suppression_reason": extracted["suppression_reason"],
            "probability_contract_json": _safe_json_dumps(extracted["probability_contract_json"]),
            "model_hash": p.get("model_hash"),
            "is_mock": bool(p.get("is_mock", False)),
            "meta_json": _safe_json_dumps(meta_json, default={}),
            "decision_state": p.get("decision_state", "UNKNOWN"),
            "risk_gate_status": p.get("risk_gate_status", "UNKNOWN"),
            "data_quality_state": p.get("data_quality_state", "UNKNOWN"),
            "confidence_state": p.get("confidence_state", "UNKNOWN"),
            "blocked_reasons_json": _safe_json_dumps(p.get("blocked_reasons", []), default=[]),
            "degraded_reasons_json": _safe_json_dumps(p.get("degraded_reasons", []), default=[]),
            "validation_eligible": bool(p.get("validation_eligible", True)),
            "gate_json": _safe_json_dumps(p.get("gate_json", {}), default={}),
            "alignment_status": p.get("alignment_status", "UNKNOWN"),
            "source_ts_min_utc": p.get("source_ts_min_utc"),
            "source_ts_max_utc": p.get("source_ts_max_utc"),
            "critical_missing_count": int(p.get("critical_missing_count", 0)),
            "decision_window_id": p.get("decision_window_id", "UNKNOWN"),
        }

        columns = list(row.keys())
        placeholders = ", ".join(["?"] * len(columns))
        update_assignments = ",\n                    ".join(
            [f"{col} = src.{col}" for col in columns if col != "prediction_business_key"]
        )
        query = f"""
            MERGE INTO predictions AS tgt
            USING (
                SELECT {placeholders}
            ) AS src ({', '.join(columns)})
            ON tgt.prediction_business_key = src.prediction_business_key
            WHEN MATCHED THEN UPDATE SET
                    {update_assignments}
            WHEN NOT MATCHED THEN INSERT ({', '.join(columns)})
            VALUES ({', '.join([f'src.{col}' for col in columns])})
        """
        con.execute(query, [row[col] for col in columns])

        if isinstance(p, dict):
            p.setdefault("prediction_id", prediction_id)
            p.setdefault("prediction_business_key", business_key)
            p.setdefault("target_name", extracted["target_name"])
            p.setdefault("target_version", extracted["target_version"])
            p.setdefault("label_version", extracted["label_version"])
            p.setdefault("feature_version", extracted["feature_version"])
            p.setdefault("calibration_version", extracted["calibration_version"])
            p.setdefault("threshold_policy_version", extracted["threshold_policy_version"])
            p.setdefault("replay_mode", extracted["replay_mode"])
            p.setdefault("ood_state", extracted["ood_state"])
            p.setdefault("ood_reason", extracted["ood_reason"])
            p.setdefault("calibration_scope", extracted["calibration_scope"])
            p.setdefault("calibration_artifact_hash", extracted["calibration_artifact_hash"])
            p.setdefault("decision_path_contract_version", extracted["decision_path_contract_version"])
            p.setdefault("suppression_reason", extracted["suppression_reason"])

        self.insert_decision_trace(
            con,
            {**dict(p), **row, "prediction_id": prediction_id, "prediction_business_key": business_key},
        )
        return prediction_id

    def insert_decision_trace(self, con: duckdb.DuckDBPyConnection, prediction: Mapping[str, Any]) -> str:
        trace = build_prediction_trace(prediction)
        event_type = str(trace.get("suppression_reason") or "").upper()
        if event_type in ("MISSING_CALIBRATION_ARTIFACT", "INVALID_CALIBRATION_ARTIFACT", "CALIBRATION_TARGET_MISMATCH"):
            normalized_event = "calibration_missing"
        elif event_type == "OOD_REJECTION" or str(trace.get("ood_state") or "").upper() == "OUT_OF_DISTRIBUTION":
            normalized_event = "ood_rejection"
        elif trace.get("suppression_reason") not in (None, "") or str(trace.get("risk_gate_status") or "").upper() == "BLOCKED" or str(trace.get("decision_state") or "").upper() == "NO_SIGNAL":
            normalized_event = "signal_suppressed"
        elif str(trace.get("risk_gate_status") or "").upper() == "DEGRADED" or str(trace.get("data_quality_state") or "").upper() in {"PARTIAL", "DEGRADED", "STALE"}:
            normalized_event = "signal_degraded"
        else:
            normalized_event = "signal_emitted"

        business_key = str(prediction.get("prediction_business_key") or _prediction_business_key(
            str(prediction["snapshot_id"]),
            str(prediction.get("horizon_kind") or "FIXED"),
            prediction.get("horizon_minutes"),
            prediction.get("horizon_seconds"),
        ))
        trace_id = _decision_trace_id(business_key, normalized_event)
        extracted = _extract_prediction_contract_fields(prediction)
        row = {
            "trace_id": trace_id,
            "created_at_utc": datetime.now(UTC),
            "prediction_id": prediction.get("prediction_id"),
            "prediction_business_key": business_key,
            "snapshot_id": prediction.get("snapshot_id"),
            "event_type": normalized_event,
            "decision_state": trace.get("decision_state"),
            "risk_gate_status": trace.get("risk_gate_status"),
            "data_quality_state": trace.get("data_quality_state"),
            "confidence_state": trace.get("confidence_state"),
            "suppression_reason": trace.get("suppression_reason"),
            "ood_state": trace.get("ood_state"),
            "ood_reason": extracted.get("ood_reason"),
            "replay_mode": trace.get("replay_mode"),
            "model_name": trace.get("model_name"),
            "model_version": trace.get("model_version"),
            "target_name": trace.get("target_name"),
            "target_version": trace.get("target_version"),
            "calibration_version": trace.get("calibration_version"),
            "calibration_scope": _safe_json_dumps(extracted.get("calibration_scope")),
            "calibration_artifact_hash": extracted.get("calibration_artifact_hash"),
            "decision_path_contract_version": extracted.get("decision_path_contract_version"),
            "threshold_policy_version": trace.get("threshold_policy_version"),
            "blocked_reasons_json": _safe_json_dumps(trace.get("blocked_reasons"), default=[]),
            "degraded_reasons_json": _safe_json_dumps(trace.get("degraded_reasons"), default=[]),
            "trace_json": _safe_json_dumps(trace, default={}),
        }
        columns = list(row.keys())
        placeholders = ", ".join(["?"] * len(columns))
        update_assignments = ",\n                    ".join([f"{col} = src.{col}" for col in columns if col != "trace_id"])
        query = f"""
            MERGE INTO decision_traces AS tgt
            USING (
                SELECT {placeholders}
            ) AS src ({', '.join(columns)})
            ON tgt.trace_id = src.trace_id
            WHEN MATCHED THEN UPDATE SET
                    {update_assignments}
            WHEN NOT MATCHED THEN INSERT ({', '.join(columns)})
            VALUES ({', '.join([f'src.{col}' for col in columns])})
        """
        con.execute(query, [row[col] for col in columns])
        return trace_id

    def upsert_endpoint(self, con: duckdb.DuckDBPyConnection, method, path, params, registry: EndpointRegistry):
        sig = registry.signature(method, path, params)
        row = con.execute("SELECT endpoint_id FROM dim_endpoints WHERE signature=?", [sig]).fetchone()
        if row:
            return row[0]
        endpoint_id = con.execute("SELECT nextval('seq_endpoint_id')").fetchone()[0]
        con.execute(
            "INSERT INTO dim_endpoints (endpoint_id, method, path, signature, params_hash, params_json) VALUES (?,?,?,?,?,?)",
            [endpoint_id, method, path, sig, registry.params_hash(params), json.dumps(params)],
        )
        return endpoint_id

    def insert_raw_event(
        self,
        con: duckdb.DuckDBPyConnection,
        run_id,
        ticker,
        endpoint_id,
        req_at,
        rec_at,
        status,
        lat,
        ph,
        pj,
        retry,
        etype,
        emsg,
        circ,
        *,
        source_publish_time_utc: Any = None,
        source_revision: Optional[str] = None,
    ) -> str:
        safe_req = to_utc_dt(req_at, fallback=datetime.now(UTC))
        safe_rec = to_utc_dt(rec_at, fallback=safe_req)
        inferred_publish, inferred_revision = _infer_source_lineage_from_payload(pj)
        publish_ts = _coerce_optional_utc_dt(source_publish_time_utc) or inferred_publish
        revision = source_revision or inferred_revision
        event_id = uuid.uuid4()
        con.execute(
            """
            INSERT INTO raw_http_events (
                event_id, run_id, requested_at_utc, received_at_utc, ticker,
                endpoint_id, http_status, latency_ms, payload_hash, payload_json,
                source_publish_time_utc, source_revision,
                is_retry, error_type, error_msg, circuit_state_json
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            [
                str(event_id),
                str(run_id) if run_id is not None else None,
                safe_req,
                safe_rec,
                ticker,
                endpoint_id,
                status,
                lat,
                ph,
                json.dumps(pj) if pj is not None else None,
                publish_ts,
                revision,
                retry,
                etype,
                emsg,
                json.dumps(circ) if circ else None,
            ],
        )
        return str(event_id)

    def begin_run(
        self,
        con: duckdb.DuckDBPyConnection,
        asof_ts_utc,
        session_label,
        is_trading_day,
        is_early_close,
        config_version,
        api_catalog_hash,
        notes: str = "",
    ) -> str:
        run_id = uuid.uuid4()
        con.execute(
            """
            INSERT INTO meta_runs (
                run_id, started_at_utc, ended_at_utc, asof_ts_utc, session_label,
                is_trading_day, is_early_close, config_version, api_catalog_hash, notes
            ) VALUES (?,?,?,?,?,?,?,?,?,?)
            """,
            [
                str(run_id),
                datetime.now(UTC),
                None,
                asof_ts_utc,
                session_label,
                is_trading_day,
                is_early_close,
                config_version,
                api_catalog_hash,
                notes,
            ],
        )
        return str(run_id)

    def insert_config(self, con: duckdb.DuckDBPyConnection, yaml_text: str) -> int:
        config_hash = hashlib.sha256(yaml_text.encode()).hexdigest()
        config_version = con.execute("SELECT nextval('seq_config_version')").fetchone()[0]
        con.execute(
            "INSERT INTO meta_config (config_version, config_hash, config_yaml, created_at_utc) VALUES (?,?,?,?)",
            [config_version, config_hash, yaml_text, datetime.now(UTC)],
        )
        con.execute(
            "INSERT INTO config_history (config_version, ingested_at_utc, yaml_content) VALUES (?,?,?)",
            [str(config_version), datetime.now(UTC), yaml_text],
        )
        return int(config_version)

    def upsert_tickers(self, con: duckdb.DuckDBPyConnection, tickers: Sequence[str]) -> None:
        con.executemany(
            "INSERT OR IGNORE INTO dim_tickers (ticker) VALUES (?)",
            [[str(t).upper()] for t in tickers],
        )

    def get_endpoint_state(self, con: duckdb.DuckDBPyConnection, ticker: str, endpoint_id: int) -> Optional[EndpointStateRow]:
        row = con.execute(
            """
            SELECT last_success_event_id, last_success_ts_utc, last_payload_hash,
                   last_change_ts_utc, last_change_event_id
            FROM endpoint_state
            WHERE ticker=? AND endpoint_id=?
            """,
            [ticker, endpoint_id],
        ).fetchone()
        if not row:
            return None
        return EndpointStateRow(
            last_success_event_id=str(row[0]) if row[0] else None,
            last_success_ts_utc=row[1],
            last_payload_hash=row[2],
            last_change_ts_utc=row[3],
            last_change_event_id=str(row[4]) if row[4] else None,
        )

    def upsert_endpoint_state(
        self,
        con: duckdb.DuckDBPyConnection,
        ticker: str,
        endpoint_id: int,
        event_id: str,
        res: Any,
        attempt_ts_utc: datetime,
        is_success_class: bool,
        changed: bool,
    ) -> None:
        con.execute("INSERT OR IGNORE INTO endpoint_state (ticker, endpoint_id) VALUES (?,?)", [ticker, endpoint_id])
        con.execute(
            """
            UPDATE endpoint_state
            SET last_attempt_event_id=?,
                last_attempt_ts_utc=?,
                last_attempt_http_status=?,
                last_attempt_error_type=?,
                last_attempt_error_msg=?
            WHERE ticker=? AND endpoint_id=?
            """,
            [str(event_id), attempt_ts_utc, res.status_code, res.error_type, res.error_message, ticker, endpoint_id],
        )
        if is_success_class:
            con.execute(
                """
                UPDATE endpoint_state
                SET last_success_event_id=?,
                    last_success_ts_utc=?,
                    last_payload_hash=?,
                    last_change_ts_utc=CASE WHEN ? THEN ? ELSE last_change_ts_utc END,
                    last_change_event_id=CASE WHEN ? THEN ? ELSE last_change_event_id END
                WHERE ticker=? AND endpoint_id=?
                """,
                [
                    str(event_id),
                    attempt_ts_utc,
                    res.payload_hash,
                    changed,
                    attempt_ts_utc,
                    changed,
                    str(event_id),
                    ticker,
                    endpoint_id,
                ],
            )

    def insert_features(self, con: duckdb.DuckDBPyConnection, snapshot_id, features_with_meta: List[Dict[str, Any]]) -> None:
        for feature in features_with_meta:
            con.execute(
                """
                INSERT INTO features (snapshot_id, feature_key, feature_value, meta_json)
                VALUES (?,?,?,?)
                ON CONFLICT (snapshot_id, feature_key)
                DO UPDATE SET feature_value = excluded.feature_value, meta_json = excluded.meta_json
                """,
                [str(snapshot_id), feature["feature_key"], feature["feature_value"], json.dumps(feature["meta_json"])],
            )

    def insert_levels(self, con: duckdb.DuckDBPyConnection, snapshot_id, levels: List[Dict[str, Any]]) -> None:
        if not levels:
            return
        for level in levels:
            con.execute(
                """
                INSERT INTO derived_levels (snapshot_id, level_type, price, magnitude, meta_json)
                VALUES (?,?,?,?,?)
                ON CONFLICT (snapshot_id, level_type, price)
                DO UPDATE SET magnitude = excluded.magnitude, meta_json = excluded.meta_json
                """,
                [
                    str(snapshot_id),
                    level["level_type"],
                    level["price"],
                    level.get("magnitude"),
                    json.dumps(level.get("meta_json", {})),
                ],
            )

    def insert_lineage(
        self,
        con: duckdb.DuckDBPyConnection,
        snapshot_id,
        endpoint_id,
        used_event_id,
        freshness_state,
        data_age_seconds,
        payload_class,
        na_reason,
        meta_json,
    ) -> None:
        con.execute(
            """
            INSERT INTO snapshot_lineage (
                snapshot_id, endpoint_id, used_event_id, freshness_state,
                data_age_seconds, payload_class, na_reason, meta_json
            ) VALUES (?,?,?,?,?,?,?,?)
            ON CONFLICT (snapshot_id, endpoint_id)
            DO UPDATE SET
                used_event_id = excluded.used_event_id,
                freshness_state = excluded.freshness_state,
                data_age_seconds = excluded.data_age_seconds,
                payload_class = excluded.payload_class,
                na_reason = excluded.na_reason,
                meta_json = excluded.meta_json
            """,
            [
                str(snapshot_id),
                endpoint_id,
                str(used_event_id) if used_event_id else None,
                freshness_state,
                data_age_seconds,
                payload_class,
                na_reason,
                json.dumps(meta_json) if meta_json else "{}",
            ],
        )

    def end_run(self, con: duckdb.DuckDBPyConnection, run_id) -> None:
        con.execute("UPDATE meta_runs SET ended_at_utc=? WHERE run_id=?", [datetime.now(UTC), str(run_id)])

    def ro_connect(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(self.duckdb_path, read_only=True)

    def get_validation_checksum(self, con: duckdb.DuckDBPyConnection) -> str:
        rows = con.execute(
            """
            SELECT prediction_id, brier_score, log_loss, is_correct
            FROM predictions
            WHERE outcome_realized = TRUE
            ORDER BY prediction_id
            """
        ).fetchall()
        return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()

    def get_pipeline_diagnostics(self, con: duckdb.DuckDBPyConnection) -> Dict[str, Any]:
        preds = con.execute("SELECT risk_gate_status, COUNT(*) FROM predictions GROUP BY risk_gate_status").fetchall()
        lineage = con.execute("SELECT freshness_state, COUNT(*) FROM snapshot_lineage GROUP BY freshness_state").fetchall()
        traces: List[Tuple[Any, Any]] = []
        try:
            tables = {str(row[0]) for row in con.execute("SHOW TABLES").fetchall()}
            if "decision_traces" in tables:
                traces = con.execute("SELECT event_type, COUNT(*) FROM decision_traces GROUP BY event_type").fetchall()
        except Exception:
            traces = []
        return {
            "predictions_by_gate": {row[0]: row[1] for row in preds},
            "lineage_by_freshness": {row[0]: row[1] for row in lineage},
            "decision_traces_by_event": {row[0]: row[1] for row in traces},
        }
