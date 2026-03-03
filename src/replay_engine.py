from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import duckdb

from . import features as feat
from .config_loader import load_yaml
from .endpoint_truth import EndpointContext
from .ingest_engine import _validate_config, generate_predictions
from .models import ReplayMode
from .scheduler import coerce_session_state
from .storage import DbWriter, extract_prediction_contract_fields

UTC = timezone.utc
CONTRACT_FIELDS: Tuple[str, ...] = (
    "model_name",
    "model_version",
    "feature_version",
    "calibration_version",
    "threshold_policy_version",
    "target_name",
    "target_version",
    "label_version",
)

EXPLICIT_RESTATED_FLAG_KEYS: Tuple[str, ...] = (
    "replay_source_mode",
    "observation_mode",
    "data_observation_mode",
    "source_mode",
    "restated",
    "is_restated",
    "backfilled",
    "is_backfilled",
    "research_only",
    "historical_backfill",
)
RESTATED_MARKERS: Tuple[str, ...] = (
    "RESTATED",
    "BACKFILLED",
    "BACKFILL",
    "RESEARCH_RESTATED",
    "RESEARCH",
)


@dataclass(frozen=True)
class ReplayContract:
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    feature_version: Optional[str] = None
    calibration_version: Optional[str] = None
    threshold_policy_version: Optional[str] = None
    target_name: Optional[str] = None
    target_version: Optional[str] = None
    label_version: Optional[str] = None

    def to_dict(self) -> Dict[str, Optional[str]]:
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "feature_version": self.feature_version,
            "calibration_version": self.calibration_version,
            "threshold_policy_version": self.threshold_policy_version,
            "target_name": self.target_name,
            "target_version": self.target_version,
            "label_version": self.label_version,
        }

    @classmethod
    def from_mapping(cls, raw: Optional[Mapping[str, Any]]) -> "ReplayContract":
        src = dict(raw or {})
        normalized: Dict[str, Optional[str]] = {}
        for field in CONTRACT_FIELDS:
            value = src.get(field)
            normalized[field] = None if value in (None, "") else str(value)
        return cls(**normalized)


@dataclass(frozen=True)
class ReplayRunReport:
    replay_run_id: str
    ticker: str
    replay_mode: ReplayMode
    snapshot_count: int
    prediction_count: int
    frozen_contract: ReplayContract
    recomputed_predictions: Tuple[Dict[str, Any], ...]
    status: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "replay_run_id": self.replay_run_id,
            "ticker": self.ticker,
            "replay_mode": self.replay_mode.value,
            "snapshot_count": self.snapshot_count,
            "prediction_count": self.prediction_count,
            "frozen_contract": self.frozen_contract.to_dict(),
            "recomputed_predictions": [dict(p) for p in self.recomputed_predictions],
            "status": self.status,
        }


def _coerce_replay_mode(value: Any) -> ReplayMode:
    if isinstance(value, ReplayMode):
        if value == ReplayMode.UNKNOWN:
            return ReplayMode.LIVE_LIKE_OBSERVED
        return value
    if value in (None, ""):
        return ReplayMode.LIVE_LIKE_OBSERVED
    try:
        mode = ReplayMode(str(value).upper())
    except Exception as exc:  # pragma: no cover - defensive input guard
        raise ValueError(f"Unsupported replay_mode: {value}") from exc
    return ReplayMode.LIVE_LIKE_OBSERVED if mode == ReplayMode.UNKNOWN else mode


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


def _ensure_utc(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            return None
        try:
            return datetime.fromtimestamp(float(value), UTC)
        except (OverflowError, OSError, ValueError):
            return None
    if isinstance(value, str):
        raw = value.strip()
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


def _table_columns(con: duckdb.DuckDBPyConnection, table: str) -> set[str]:
    return {str(row[1]) for row in con.execute(f"PRAGMA table_info('{table}')").fetchall()}


def _select_expr(columns: set[str], column: str, alias: Optional[str] = None) -> str:
    alias = alias or column
    return column if column in columns else f"NULL AS {alias}"


def _fetch_raw_event_map(
    con: duckdb.DuckDBPyConnection,
    event_ids: Sequence[str],
) -> Dict[str, Dict[str, Any]]:
    if not event_ids:
        return {}
    raw_cols = _table_columns(con, "raw_http_events")
    placeholders = ",".join(["?"] * len(event_ids))
    query = f"""
        SELECT event_id,
               payload_json,
               {_select_expr(raw_cols, 'requested_at_utc')},
               {_select_expr(raw_cols, 'received_at_utc')},
               {_select_expr(raw_cols, 'source_publish_time_utc')},
               {_select_expr(raw_cols, 'source_revision')}
        FROM raw_http_events
        WHERE event_id IN ({placeholders})
    """
    rows = con.execute(query, list(event_ids)).fetchall()
    out: Dict[str, Dict[str, Any]] = {}
    for event_id, payload_json, requested_at_utc, received_at_utc, source_publish_time_utc, source_revision in rows:
        payload: Any = payload_json
        if isinstance(payload_json, str):
            try:
                payload = json.loads(payload_json)
            except Exception:
                payload = None
        out[str(event_id)] = {
            "payload": payload,
            "requested_at_utc": requested_at_utc,
            "received_at_utc": received_at_utc,
            "source_publish_time_utc": source_publish_time_utc,
            "source_revision": source_revision,
        }
    return out


def _fetch_endpoint_map(con: duckdb.DuckDBPyConnection, endpoint_ids: Iterable[int]) -> Dict[int, Tuple[Any, Any, Any]]:
    ids = sorted({int(eid) for eid in endpoint_ids if eid is not None})
    if not ids:
        return {}
    placeholders = ",".join(["?"] * len(ids))
    rows = con.execute(
        f"SELECT endpoint_id, method, path, signature FROM dim_endpoints WHERE endpoint_id IN ({placeholders})",
        ids,
    ).fetchall()
    return {int(eid): (method, path, signature) for eid, method, path, signature in rows}


def _fetch_predictions(
    con: duckdb.DuckDBPyConnection,
    *,
    snapshot_ids: Optional[Sequence[str]] = None,
    snapshot_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    pred_cols = _table_columns(con, "predictions")
    select_columns = [
        "prediction_id",
        "snapshot_id",
        "horizon_kind",
        "horizon_minutes",
        "horizon_seconds",
        "decision_state",
        "risk_gate_status",
        "prob_up",
        "prob_down",
        "prob_flat",
        "model_name",
        "model_version",
        "feature_version",
        "calibration_version",
        "threshold_policy_version",
        "target_name",
        "target_version",
        "label_version",
        "replay_mode",
        "meta_json",
    ]
    select_sql = ", ".join(_select_expr(pred_cols, col) for col in select_columns)

    where_sql = ""
    params: List[Any] = []
    if snapshot_id is not None:
        where_sql = "WHERE snapshot_id = ?"
        params = [snapshot_id]
    elif snapshot_ids is not None:
        ids = [str(x) for x in snapshot_ids]
        if not ids:
            return []
        placeholders = ",".join(["?"] * len(ids))
        where_sql = f"WHERE snapshot_id IN ({placeholders})"
        params = ids

    rows = con.execute(f"SELECT {select_sql} FROM predictions {where_sql}", params).fetchall()
    return [dict(zip(select_columns, row)) for row in rows]


def _contract_from_prediction(prediction: Mapping[str, Any]) -> Dict[str, Optional[str]]:
    extracted = extract_prediction_contract_fields(prediction)
    contract = {
        "model_name": None if prediction.get("model_name") in (None, "") else str(prediction.get("model_name")),
        "model_version": None if prediction.get("model_version") in (None, "") else str(prediction.get("model_version")),
        "feature_version": extracted.get("feature_version"),
        "calibration_version": extracted.get("calibration_version"),
        "threshold_policy_version": extracted.get("threshold_policy_version"),
        "target_name": extracted.get("target_name"),
        "target_version": extracted.get("target_version"),
        "label_version": extracted.get("label_version"),
        "replay_mode": extracted.get("replay_mode") or (None if prediction.get("replay_mode") in (None, "") else str(prediction.get("replay_mode"))),
    }
    return contract


def _merge_frozen_contract(
    frozen: Dict[str, Optional[str]],
    observed: Mapping[str, Optional[str]],
    *,
    context: str,
) -> Dict[str, Optional[str]]:
    merged = dict(frozen)
    for field in CONTRACT_FIELDS:
        observed_value = observed.get(field)
        if observed_value in (None, ""):
            continue
        current = merged.get(field)
        if current in (None, ""):
            merged[field] = str(observed_value)
        elif str(current) != str(observed_value):
            raise RuntimeError(
                f"Artifact version drift inside replay run: {field} differs within {context} ({current} != {observed_value})"
            )
    return merged


def _enforce_stored_contract_invariants(
    stored_predictions: Sequence[Mapping[str, Any]],
    *,
    replay_mode: ReplayMode,
) -> Dict[str, Optional[str]]:
    frozen: Dict[str, Optional[str]] = {field: None for field in CONTRACT_FIELDS}
    explicit_modes: set[str] = set()

    for pred in stored_predictions:
        observed = _contract_from_prediction(pred)
        frozen = _merge_frozen_contract(frozen, observed, context="stored_predictions")
        mode = observed.get("replay_mode")
        if mode not in (None, "", ReplayMode.UNKNOWN.value):
            explicit_modes.add(str(mode))

    if len(explicit_modes) > 1:
        raise RuntimeError(f"Mixed replay modes are invalid within a replay run: {sorted(explicit_modes)}")
    if len(explicit_modes) == 1 and replay_mode.value not in explicit_modes:
        raise RuntimeError(
            f"Replay mode mismatch: stored predictions are stamped {sorted(explicit_modes)[0]} but run requested {replay_mode.value}"
        )
    return frozen


def _extract_explicit_restated_violations(
    details: Mapping[str, Any],
    raw_info: Mapping[str, Any],
    payload: Any,
    asof_ts: datetime,
) -> List[str]:
    violations: List[str] = []

    sources: List[Mapping[str, Any]] = [details, raw_info]
    if isinstance(payload, Mapping):
        sources.append(payload)

    for source in sources:
        for key in EXPLICIT_RESTATED_FLAG_KEYS:
            if key not in source:
                continue
            value = source.get(key)
            if value is True:
                violations.append(f"explicit_restated_flag:{key}")
                continue
            if isinstance(value, str):
                text = value.strip().upper()
                if any(marker in text for marker in RESTATED_MARKERS):
                    violations.append(f"explicit_restated_mode:{key}={text}")

    publish_candidates = [
        details.get("source_publish_time_utc"),
        raw_info.get("source_publish_time_utc"),
        details.get("event_time_utc"),
        details.get("effective_ts_utc"),
    ]
    for label, candidate in (
        ("source_publish_after_snapshot", publish_candidates[0]),
        ("raw_event_publish_after_snapshot", publish_candidates[1]),
        ("event_time_after_snapshot", publish_candidates[2]),
        ("effective_time_after_snapshot", publish_candidates[3]),
        ("received_after_snapshot", raw_info.get("received_at_utc")),
        ("requested_after_snapshot", raw_info.get("requested_at_utc")),
    ):
        ts = _ensure_utc(candidate)
        if ts is not None and ts > asof_ts:
            violations.append(label)

    revision = details.get("source_revision") or raw_info.get("source_revision")
    if isinstance(revision, str):
        rev_text = revision.strip().upper()
        if any(marker in rev_text for marker in RESTATED_MARKERS):
            violations.append("revision_marks_restated")

    return sorted(set(violations))


def _stamp_replay_prediction(
    prediction: Mapping[str, Any],
    *,
    replay_run_id: str,
    replay_mode: ReplayMode,
    frozen_contract: Mapping[str, Optional[str]],
) -> Dict[str, Any]:
    stamped = dict(prediction)
    stamped["replay_mode"] = replay_mode.value
    for field in CONTRACT_FIELDS:
        if field in frozen_contract and frozen_contract.get(field) not in (None, ""):
            stamped[field] = frozen_contract.get(field)

    meta_json = _safe_json_loads(stamped.get("meta_json"))
    meta_json["replay_mode"] = replay_mode.value
    meta_json["replay_run_id"] = replay_run_id
    meta_json["replay_contract"] = dict(frozen_contract)
    stamped["meta_json"] = meta_json
    return stamped


def run_replay(
    db_path: str,
    ticker: str,
    start_ts: Optional[str] = None,
    end_ts: Optional[str] = None,
    cfg: Optional[Dict[str, Any]] = None,
    *,
    replay_mode: Any = ReplayMode.LIVE_LIKE_OBSERVED,
) -> Dict[str, Any]:
    """
    Replay predictions using the exact live scoring path while making replay mode explicit.
    LIVE_LIKE_OBSERVED forbids restated/backfilled or not-yet-observable data.
    RESEARCH_RESTATED allows historically restated data but still freezes the artifact contract.
    """
    if cfg is None:
        cfg = load_yaml("src/config/config.yaml").raw
    _validate_config(cfg)

    requested_mode = _coerce_replay_mode(replay_mode)
    writer = DbWriter(db_path)
    con = duckdb.connect(db_path)
    writer.ensure_replay_schema(con)

    started_at_utc = datetime.now(UTC)
    requested_start_dt = _ensure_utc(start_ts)
    requested_end_dt = _ensure_utc(end_ts)

    replay_run_id = writer.begin_replay_run(
        con,
        ticker=ticker,
        replay_mode=requested_mode.value,
        started_at_utc=started_at_utc,
        requested_start_ts_utc=requested_start_dt,
        requested_end_ts_utc=requested_end_dt,
        contract={},
    )

    snapshot_count = 0
    prediction_count = 0
    stored_contract: Dict[str, Optional[str]] = {field: None for field in CONTRACT_FIELDS}
    frozen_contract: Dict[str, Optional[str]] = {field: None for field in CONTRACT_FIELDS}
    recomputed_predictions: List[Dict[str, Any]] = []

    try:
        print(f"--- REPLAY PARITY CHECK: {ticker.upper()} [{requested_mode.value}] ---")
        query = "SELECT snapshot_id, asof_ts_utc, data_quality_score, session_label, seconds_to_close FROM snapshots WHERE ticker = ?"
        params: List[Any] = [ticker.upper()]
        if start_ts:
            query += " AND asof_ts_utc >= ?"
            params.append(start_ts)
        if end_ts:
            query += " AND asof_ts_utc <= ?"
            params.append(end_ts)
        query += " ORDER BY asof_ts_utc ASC"

        snapshots = con.execute(query, params).fetchall()
        snapshot_ids = [str(row[0]) for row in snapshots]

        stored_predictions_all = _fetch_predictions(con, snapshot_ids=snapshot_ids)
        stored_contract = _enforce_stored_contract_invariants(stored_predictions_all, replay_mode=requested_mode)

        for snap_id, asof_ts, dq, sess_str, sec_to_close in snapshots:
            snapshot_count += 1
            asof_ts = _ensure_utc(asof_ts) or datetime.now(UTC)
            stored_preds = _fetch_predictions(con, snapshot_id=str(snap_id))

            lineage_rows = con.execute(
                "SELECT endpoint_id, used_event_id, freshness_state, data_age_seconds, na_reason, payload_class, meta_json FROM snapshot_lineage WHERE snapshot_id = ?",
                [str(snap_id)],
            ).fetchall()
            raw_event_map = _fetch_raw_event_map(con, [str(r[1]) for r in lineage_rows if r[1] is not None])
            endpoint_map = _fetch_endpoint_map(con, [int(r[0]) for r in lineage_rows])

            effective_payloads: Dict[int, Any] = {}
            contexts: Dict[int, EndpointContext] = {}

            for endpoint_id, used_event_id, freshness_state, age_seconds, na_reason, payload_class, meta_json_val in lineage_rows:
                meta_dict = _safe_json_loads(meta_json_val)
                details = meta_dict.get("details") if isinstance(meta_dict.get("details"), dict) else {}
                raw_info = raw_event_map.get(str(used_event_id), {}) if used_event_id is not None else {}
                payload = raw_info.get("payload") if raw_info else None

                if requested_mode == ReplayMode.LIVE_LIKE_OBSERVED:
                    observability_violations = _extract_explicit_restated_violations(details, raw_info, payload, asof_ts)
                    if observability_violations:
                        raise RuntimeError(
                            f"LIVE_LIKE_OBSERVED replay contaminated by restated/backfilled data: "
                            f"snapshot {snap_id} endpoint {endpoint_id} violations={observability_violations}"
                        )

                method, path, signature = endpoint_map.get(int(endpoint_id), (None, None, None))
                if method is None or path is None or signature is None:
                    continue

                effective_payloads[int(endpoint_id)] = payload
                if used_event_id and payload is None:
                    na_reason = "missing_raw_payload_for_lineage"

                missing_field = None
                if freshness_state in ("FRESH", "STALE_CARRY", "EMPTY_VALID"):
                    if "effective_ts_utc" not in details:
                        missing_field = "effective_ts_utc"
                    elif "endpoint_asof_ts_utc" not in details:
                        missing_field = "endpoint_asof_ts_utc"
                    elif "truth_status" not in details:
                        missing_field = "truth_status"
                    elif freshness_state == "STALE_CARRY" and "stale_age_seconds" not in details:
                        missing_field = "stale_age_seconds"

                effective_ts_utc = _ensure_utc(details.get("effective_ts_utc"))
                endpoint_asof_utc = _ensure_utc(details.get("endpoint_asof_ts_utc")) or asof_ts
                event_time_utc = _ensure_utc(details.get("event_time_utc"))
                publish_time_utc = _ensure_utc(details.get("source_publish_time_utc")) or _ensure_utc(raw_info.get("source_publish_time_utc"))
                received_at_utc = _ensure_utc(details.get("received_at_utc")) or _ensure_utc(raw_info.get("received_at_utc"))
                processed_at_utc = _ensure_utc(details.get("processed_at_utc"))
                as_of_time_utc = _ensure_utc(details.get("as_of_time_utc")) or asof_ts
                source_revision = details.get("source_revision") or raw_info.get("source_revision")
                effective_time_source = details.get("effective_time_source")
                timestamp_quality = details.get("timestamp_quality")
                lagged = bool(details.get("lagged", False))
                provenance_degraded = bool(details.get("time_provenance_degraded", False))

                if missing_field:
                    freshness_state = "ERROR"
                    na_reason = f"replay_missing_lineage_field:{missing_field}"
                    effective_ts_utc = None
                    endpoint_asof_utc = asof_ts
                    provenance_degraded = True

                delta_sec = details.get("alignment_delta_sec", age_seconds if age_seconds is not None else 0)
                stale_age_min = None if age_seconds is None else int(float(age_seconds) // 60)
                contexts[int(endpoint_id)] = EndpointContext(
                    endpoint_id=int(endpoint_id),
                    method=method,
                    path=path,
                    operation_id=None,
                    signature=signature,
                    used_event_id=str(used_event_id) if used_event_id is not None else None,
                    payload_class=str(payload_class),
                    freshness_state=str(freshness_state),
                    stale_age_min=stale_age_min,
                    na_reason=None if na_reason in (None, "") else str(na_reason),
                    endpoint_asof_ts_utc=endpoint_asof_utc,
                    alignment_delta_sec=None if delta_sec is None else int(delta_sec),
                    effective_ts_utc=effective_ts_utc,
                    event_time_utc=event_time_utc,
                    source_publish_time_utc=publish_time_utc,
                    received_at_utc=received_at_utc,
                    processed_at_utc=processed_at_utc,
                    as_of_time_utc=as_of_time_utc,
                    source_revision=None if source_revision in (None, "") else str(source_revision),
                    effective_time_source=None if effective_time_source in (None, "") else str(effective_time_source),
                    timestamp_quality=None if timestamp_quality in (None, "") else str(timestamp_quality),
                    lagged=lagged,
                    time_provenance_degraded=provenance_degraded,
                )

            feature_rows, _ = feat.extract_all(effective_payloads, contexts)
            valid_features: List[Dict[str, Any]] = []
            for feature_row in feature_rows:
                if not (isinstance(feature_row, dict) and "feature_key" in feature_row and "meta_json" in feature_row):
                    continue
                feature_value = feature_row.get("feature_value")
                if feature_value is not None and not math.isfinite(feature_value):
                    continue
                valid_features.append(feature_row)

            session_enum = coerce_session_state(sess_str)
            snapshot_predictions = generate_predictions(
                cfg=cfg,
                snapshot_id=snap_id,
                valid_features=valid_features,
                asof_utc=asof_ts,
                session_enum=session_enum,
                sec_to_close=sec_to_close,
                endpoint_coverage=dq,
            )

            stored_pred_map = {
                (row.get("horizon_kind"), row.get("horizon_minutes"), row.get("horizon_seconds")): row
                for row in stored_preds
            }
            replay_pred_map: Dict[Tuple[Any, Any, Any], Dict[str, Any]] = {}
            for pred in snapshot_predictions:
                observed_contract = _contract_from_prediction(pred)
                frozen_contract = _merge_frozen_contract(
                    frozen_contract,
                    observed_contract,
                    context=f"recomputed_predictions snapshot={snap_id}",
                )
                stamped = _stamp_replay_prediction(
                    pred,
                    replay_run_id=replay_run_id,
                    replay_mode=requested_mode,
                    frozen_contract=frozen_contract,
                )
                replay_pred_map[(stamped["horizon_kind"], stamped["horizon_minutes"], stamped["horizon_seconds"])] = stamped
                recomputed_predictions.append(stamped)
                prediction_count += 1

            if stored_preds:
                for sp_key, stored_row in stored_pred_map.items():
                    if sp_key not in replay_pred_map:
                        raise RuntimeError(
                            f"PARITY MISMATCH: Snapshot {snap_id} Horizon {sp_key} exists in stored but replay could not compute it."
                        )
                    replay_row = replay_pred_map[sp_key]
                    if stored_row.get("decision_state") != replay_row["decision_state"] or stored_row.get("risk_gate_status") != replay_row["risk_gate_status"]:
                        raise RuntimeError(
                            f"PARITY MISMATCH: Snapshot {snap_id} Horizon {sp_key} Risk/Decision Governance altered. "
                            f"Stored: {stored_row.get('decision_state')}/{stored_row.get('risk_gate_status')} | "
                            f"Recomputed: {replay_row['decision_state']}/{replay_row['risk_gate_status']}"
                        )

        final_contract_payload = dict(frozen_contract)
        if all(value in (None, "") for value in final_contract_payload.values()):
            final_contract_payload = dict(stored_contract)
        frozen_contract_obj = ReplayContract.from_mapping(final_contract_payload)
        writer.finish_replay_run(
            con,
            replay_run_id,
            finished_at_utc=datetime.now(UTC),
            status="PASSED",
            snapshot_count=snapshot_count,
            prediction_count=prediction_count,
            contract=frozen_contract_obj.to_dict(),
        )
        print(
            "✅ Replay Parity Check Passed: "
            f"mode={requested_mode.value} run_id={replay_run_id} snapshots={snapshot_count} predictions={prediction_count}"
        )
        report = ReplayRunReport(
            replay_run_id=replay_run_id,
            ticker=ticker.upper(),
            replay_mode=requested_mode,
            snapshot_count=snapshot_count,
            prediction_count=prediction_count,
            frozen_contract=frozen_contract_obj,
            recomputed_predictions=tuple(recomputed_predictions),
            status="PASSED",
        )
        return report.to_dict()
    except Exception as exc:
        final_contract_payload = dict(frozen_contract)
        if all(value in (None, "") for value in final_contract_payload.values()):
            final_contract_payload = dict(stored_contract)
        writer.finish_replay_run(
            con,
            replay_run_id,
            finished_at_utc=datetime.now(UTC),
            status="FAILED",
            snapshot_count=snapshot_count,
            prediction_count=prediction_count,
            failure_reason=str(exc),
            contract=ReplayContract.from_mapping(final_contract_payload).to_dict(),
        )
        raise
    finally:
        con.close()
