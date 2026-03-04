from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import duckdb
import pandas as pd

try:  # pragma: no cover - import is environment-dependent
    import streamlit as st
except Exception:  # pragma: no cover - import is environment-dependent
    st = None  # type: ignore[assignment]

from src.config_loader import load_yaml

PAGE_TITLE = "Decision Support Dashboard"
DASHBOARD_DISCLAIMER = (
    "Analytics/reporting only. This dashboard does not imply live-trade readiness. "
    "Governance state is shown explicitly. Probability columns are displayed only when the stored contract is coherent, "
    "unsuppressed, and not in an unknown-governance state."
)
ET = ZoneInfo("America/New_York")

PredictionField = Tuple[str, Optional[str]]

PREDICTION_REPORT_FIELDS: Sequence[PredictionField] = (
    ("horizon_minutes", None),
    ("horizon_kind", None),
    ("horizon_seconds", None),
    ("decision_state", None),
    ("risk_gate_status", None),
    ("data_quality_state", "quality_state"),
    ("confidence_state", None),
    ("target_name", None),
    ("target_version", None),
    ("label_version", None),
    ("model_name", None),
    ("model_version", None),
    ("calibration_version", None),
    ("threshold_policy_version", None),
    ("replay_mode", None),
    ("ood_state", None),
    ("ood_reason", None),
    ("calibration_scope", None),
    ("calibration_artifact_hash", None),
    ("decision_path_contract_version", None),
    ("suppression_reason", None),
    ("prob_up", None),
    ("prob_down", None),
    ("prob_flat", None),
    ("confidence", None),
    ("probability_contract_json", None),
    ("meta_json", None),
)

DECISION_TRACE_FIELDS: Sequence[PredictionField] = (
    ("event_type", None),
    ("decision_state", None),
    ("risk_gate_status", None),
    ("data_quality_state", "quality_state"),
    ("confidence_state", None),
    ("suppression_reason", None),
    ("ood_state", None),
    ("ood_reason", None),
    ("replay_mode", None),
    ("model_version", None),
    ("target_version", None),
    ("calibration_version", None),
    ("calibration_scope", None),
    ("calibration_artifact_hash", None),
    ("decision_path_contract_version", None),
    ("trace_json", None),
    ("created_at_utc", None),
)

REALIZED_REPORT_FIELDS: Sequence[PredictionField] = (
    ("realized_at_utc", None),
    ("brier_score", None),
    ("data_quality_state", "quality_state"),
    ("confidence_state", None),
    ("target_version", None),
    ("calibration_version", None),
    ("replay_mode", None),
    ("ood_state", None),
    ("ood_reason", None),
    ("calibration_scope", None),
    ("calibration_artifact_hash", None),
    ("decision_path_contract_version", None),
    ("suppression_reason", None),
    ("prob_up", None),
    ("prob_down", None),
    ("prob_flat", None),
    ("probability_contract_json", None),
    ("meta_json", None),
)


def _require_streamlit() -> Any:
    if st is None:
        raise RuntimeError("streamlit is required to render the dashboard UI")
    return st


def configure_page() -> None:
    ui = _require_streamlit()
    ui.set_page_config(layout="wide", page_title=PAGE_TITLE)


def get_db() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(load_yaml("src/config/config.yaml").raw["storage"]["duckdb_path"], read_only=True)


def _existing_columns(con: duckdb.DuckDBPyConnection, table: str) -> set[str]:
    return {str(r[1]) for r in con.execute(f"PRAGMA table_info('{table}')").fetchall()}


def _has_column(con: duckdb.DuckDBPyConnection, table: str, col: str) -> bool:
    return col in _existing_columns(con, table)


def _has_table(con: duckdb.DuckDBPyConnection, table: str) -> bool:
    rows = con.execute("SHOW TABLES").fetchall()
    return table in {str(r[0]) for r in rows}


def _column_expr(
    con: duckdb.DuckDBPyConnection,
    *,
    table: str,
    sql_alias: str,
    column: str,
    alias: Optional[str] = None,
) -> str:
    out_name = alias or column
    if _has_column(con, table, column):
        return f"{sql_alias}.{column} AS {out_name}"
    return f"NULL AS {out_name}"


def _json_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except Exception:
            return {}
        return dict(parsed) if isinstance(parsed, Mapping) else {}
    return {}


def _json_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except Exception:
            return []
        if isinstance(parsed, list):
            return list(parsed)
    return []


def _clean_optional_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    text = str(value).strip()
    return text if text else None


def _vector_component(vector: Any, key: str) -> Optional[float]:
    if isinstance(vector, Mapping):
        value = vector.get(key)
    elif isinstance(vector, (list, tuple)):
        index = {"UP": 0, "DOWN": 1, "FLAT": 2}.get(key)
        value = vector[index] if index is not None and index < len(vector) else None
    else:
        value = None
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _resolve_probability_contract(row: Mapping[str, Any]) -> Dict[str, Any]:
    contract = _json_dict(row.get("probability_contract_json"))
    if contract:
        return contract
    trace_json = _json_dict(row.get("trace_json"))
    return _json_dict(trace_json.get("probability_contract"))


def _resolve_meta_json(row: Mapping[str, Any]) -> Dict[str, Any]:
    return _json_dict(row.get("meta_json"))


def _resolve_trace_json(row: Mapping[str, Any]) -> Dict[str, Any]:
    return _json_dict(row.get("trace_json"))


def _resolve_calibration_scope(row: Mapping[str, Any]) -> Dict[str, Any]:
    probability_contract = _resolve_probability_contract(row)
    calibration_ref = _json_dict(probability_contract.get("calibration_artifact_ref"))
    prediction_contract = _json_dict(_resolve_meta_json(row).get("prediction_contract"))
    trace_json = _resolve_trace_json(row)

    scope = _json_dict(row.get("calibration_scope"))
    if scope:
        return scope
    for candidate in (
        probability_contract.get("calibration_scope"),
        calibration_ref.get("calibration_scope"),
        prediction_contract.get("calibration_scope"),
        trace_json.get("calibration_scope"),
    ):
        scope = _json_dict(candidate)
        if scope:
            return scope
    return {}


def _resolve_calibration_artifact_hash(row: Mapping[str, Any]) -> Optional[str]:
    probability_contract = _resolve_probability_contract(row)
    calibration_ref = _json_dict(probability_contract.get("calibration_artifact_ref"))
    trace_json = _resolve_trace_json(row)
    return _clean_optional_text(
        row.get("calibration_artifact_hash")
        or calibration_ref.get("artifact_hash")
        or trace_json.get("calibration_artifact_hash")
    )


def _resolve_decision_path_contract_version(row: Mapping[str, Any]) -> Optional[str]:
    meta_json = _resolve_meta_json(row)
    trace_json = _resolve_trace_json(row)
    prediction_contract = _json_dict(meta_json.get("prediction_contract"))
    horizon_contract = _json_dict(meta_json.get("horizon_contract"))
    trace_horizon_contract = _json_dict(trace_json.get("horizon_contract"))
    return _clean_optional_text(
        row.get("decision_path_contract_version")
        or meta_json.get("decision_path_contract_version")
        or prediction_contract.get("decision_path_contract_version")
        or horizon_contract.get("decision_path_contract_version")
        or trace_json.get("decision_path_contract_version")
        or trace_horizon_contract.get("decision_path_contract_version")
    )


def _resolve_decision_path_exclusions(row: Mapping[str, Any]) -> Dict[str, List[str]]:
    meta_json = _resolve_meta_json(row)
    trace_json = _resolve_trace_json(row)
    horizon_contract = _json_dict(meta_json.get("horizon_contract"))
    if not horizon_contract:
        horizon_contract = _json_dict(trace_json.get("horizon_contract"))
    exclusion_fields = {
        "zero_weight": horizon_contract.get("zero_weight_excluded_features") or [],
        "report_only": horizon_contract.get("report_only_excluded_features") or [],
        "context_only": horizon_contract.get("context_only_excluded_features") or [],
        "disabled": horizon_contract.get("disabled_excluded_features") or [],
        "non_decision_eligible": horizon_contract.get("non_decision_eligible_excluded_features") or [],
        "contract_violations": horizon_contract.get("contract_violation_features") or [],
        "critical_overrides": horizon_contract.get("explicit_critical_override_features") or [],
    }
    normalized: Dict[str, List[str]] = {}
    for key, value in exclusion_fields.items():
        items = [str(v) for v in _json_list(value) if _clean_optional_text(v) is not None]
        if items:
            normalized[key] = items
    return normalized


def _summarize_decision_path_exclusions(exclusions: Mapping[str, List[str]]) -> Optional[str]:
    if not exclusions:
        return None
    parts = [f"{key}={','.join(values)}" for key, values in exclusions.items() if values]
    return " | ".join(parts) if parts else None


def _summarize_calibration_scope(scope: Mapping[str, Any]) -> Optional[str]:
    if not scope:
        return None
    pieces: List[str] = []
    target = _clean_optional_text(scope.get("target"))
    if target:
        pieces.append(f"target={target}")
    horizon_kind = _clean_optional_text(scope.get("horizon_kind"))
    horizon_minutes = scope.get("horizon_minutes")
    if horizon_kind:
        if horizon_minutes not in (None, "", "ANY"):
            pieces.append(f"horizon={horizon_kind}:{int(horizon_minutes)}")
        else:
            pieces.append(f"horizon={horizon_kind}")
    session = _clean_optional_text(scope.get("session"))
    if session:
        pieces.append(f"session={session}")
    regime = _clean_optional_text(scope.get("regime"))
    if regime:
        pieces.append(f"regime={regime}")
    replay_mode = _clean_optional_text(scope.get("replay_mode"))
    if replay_mode:
        pieces.append(f"replay={replay_mode}")
    scope_contract_version = _clean_optional_text(scope.get("scope_contract_version"))
    if scope_contract_version:
        pieces.append(f"scope_ver={scope_contract_version}")
    return " | ".join(pieces) if pieces else None


def _display_probs_for_state(state: str) -> bool:
    return state in {"CALIBRATED", "DEGRADED"}


def _derive_prediction_governance(
    row: Mapping[str, Any],
) -> Tuple[str, Optional[str], Optional[float], Optional[float], Optional[float], Dict[str, Any], Dict[str, List[str]]]:
    probability_contract = _resolve_probability_contract(row)
    calibration_ref = _json_dict(probability_contract.get("calibration_artifact_ref"))
    suppression_reason = _clean_optional_text(row.get("suppression_reason") or probability_contract.get("suppression_reason"))
    ood_state = _clean_optional_text(row.get("ood_state") or probability_contract.get("ood_state"))
    ood_reason = _clean_optional_text(row.get("ood_reason") or probability_contract.get("ood_reason"))
    target_version = _clean_optional_text(row.get("target_version"))
    calibration_version = _clean_optional_text(row.get("calibration_version") or calibration_ref.get("artifact_version"))
    replay_mode = _clean_optional_text(row.get("replay_mode"))
    quality_state = _clean_optional_text(row.get("quality_state") or row.get("data_quality_state"))
    confidence_state = _clean_optional_text(row.get("confidence_state"))
    coherent = bool(probability_contract.get("is_coherent"))
    calibrated_vector = probability_contract.get("calibrated_probability_vector")
    calibration_scope = _resolve_calibration_scope(row)
    calibration_artifact_hash = _resolve_calibration_artifact_hash(row)
    exclusions = _resolve_decision_path_exclusions(row)

    if suppression_reason is not None or ood_state == "OUT_OF_DISTRIBUTION":
        state = "SUPPRESSED"
        reason = suppression_reason or ood_reason or "OOD_REJECTION"
    elif calibrated_vector is None:
        state = "UNAVAILABLE"
        reason = "CALIBRATED_VECTOR_MISSING"
    elif not coherent:
        state = "INVALID"
        reason = "INCOHERENT_PROBABILITY_VECTOR"
    else:
        governance_gaps: List[str] = []
        if ood_state in (None, "", "UNKNOWN"):
            governance_gaps.append("OOD_UNKNOWN")
        if target_version is None:
            governance_gaps.append("TARGET_VERSION_MISSING")
        if calibration_version is None:
            governance_gaps.append("CALIBRATION_VERSION_MISSING")
        if replay_mode is None:
            governance_gaps.append("REPLAY_MODE_MISSING")
        if not calibration_scope:
            governance_gaps.append("CALIBRATION_SCOPE_MISSING")
        if calibration_artifact_hash is None:
            governance_gaps.append("CALIBRATION_ARTIFACT_HASH_MISSING")

        if governance_gaps:
            state = "UNKNOWN_GOVERNANCE"
            reason = ", ".join(governance_gaps)
        else:
            degraded_reasons: List[str] = []
            if ood_state == "DEGRADED":
                degraded_reasons.append(ood_reason or "OOD_DEGRADED")
            if quality_state and quality_state.upper() in {"PARTIAL", "DEGRADED", "STALE"}:
                degraded_reasons.append(f"QUALITY_{quality_state.upper()}")
            if confidence_state and confidence_state.upper() == "DEGRADED":
                degraded_reasons.append("CONFIDENCE_DEGRADED")
            if degraded_reasons:
                state = "DEGRADED"
                reason = ", ".join(degraded_reasons)
            else:
                state = "CALIBRATED"
                reason = None

    if _display_probs_for_state(state):
        display_up = _coerce_float(row.get("prob_up"))
        display_down = _coerce_float(row.get("prob_down"))
        display_flat = _coerce_float(row.get("prob_flat"))
        if display_up is None:
            display_up = _vector_component(calibrated_vector, "UP")
        if display_down is None:
            display_down = _vector_component(calibrated_vector, "DOWN")
        if display_flat is None:
            display_flat = _vector_component(calibrated_vector, "FLAT")
    else:
        display_up = display_down = display_flat = None

    return state, reason, display_up, display_down, display_flat, calibration_scope, exclusions


def _derive_trace_governance(row: Mapping[str, Any]) -> Tuple[str, Optional[str], Dict[str, Any], Dict[str, List[str]]]:
    suppression_reason = _clean_optional_text(row.get("suppression_reason"))
    ood_state = _clean_optional_text(row.get("ood_state"))
    ood_reason = _clean_optional_text(row.get("ood_reason"))
    calibration_version = _clean_optional_text(row.get("calibration_version"))
    replay_mode = _clean_optional_text(row.get("replay_mode"))
    quality_state = _clean_optional_text(row.get("quality_state") or row.get("data_quality_state"))
    confidence_state = _clean_optional_text(row.get("confidence_state"))
    calibration_scope = _resolve_calibration_scope(row)
    calibration_artifact_hash = _resolve_calibration_artifact_hash(row)
    exclusions = _resolve_decision_path_exclusions(row)

    if suppression_reason is not None or ood_state == "OUT_OF_DISTRIBUTION":
        return "SUPPRESSED", suppression_reason or ood_reason or "OOD_REJECTION", calibration_scope, exclusions

    governance_gaps: List[str] = []
    if ood_state in (None, "", "UNKNOWN"):
        governance_gaps.append("OOD_UNKNOWN")
    if calibration_version is None:
        governance_gaps.append("CALIBRATION_VERSION_MISSING")
    if replay_mode is None:
        governance_gaps.append("REPLAY_MODE_MISSING")
    if not calibration_scope:
        governance_gaps.append("CALIBRATION_SCOPE_MISSING")
    if calibration_artifact_hash is None:
        governance_gaps.append("CALIBRATION_ARTIFACT_HASH_MISSING")
    if governance_gaps:
        return "UNKNOWN_GOVERNANCE", ", ".join(governance_gaps), calibration_scope, exclusions

    degraded_reasons: List[str] = []
    if ood_state == "DEGRADED":
        degraded_reasons.append(ood_reason or "OOD_DEGRADED")
    if quality_state and quality_state.upper() in {"PARTIAL", "DEGRADED", "STALE"}:
        degraded_reasons.append(f"QUALITY_{quality_state.upper()}")
    if confidence_state and confidence_state.upper() == "DEGRADED":
        degraded_reasons.append("CONFIDENCE_DEGRADED")
    if degraded_reasons:
        return "DEGRADED", ", ".join(degraded_reasons), calibration_scope, exclusions

    return "CALIBRATED", None, calibration_scope, exclusions


def _normalize_prediction_contract_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    normalized = df.copy()
    if "quality_state" not in normalized.columns and "data_quality_state" in normalized.columns:
        normalized["quality_state"] = normalized["data_quality_state"]
    if "quality_state" not in normalized.columns:
        normalized["quality_state"] = "UNKNOWN"

    probability_states: List[str] = []
    governance_reasons: List[Optional[str]] = []
    calibrated_up: List[Optional[float]] = []
    calibrated_down: List[Optional[float]] = []
    calibrated_flat: List[Optional[float]] = []
    scope_dicts: List[Optional[Dict[str, Any]]] = []
    scope_labels: List[Optional[str]] = []
    artifact_hashes: List[Optional[str]] = []
    dp_versions: List[Optional[str]] = []
    exclusion_dicts: List[Optional[Dict[str, List[str]]]] = []
    exclusion_labels: List[Optional[str]] = []

    for row in normalized.to_dict(orient="records"):
        state, reason, up, down, flat, scope, exclusions = _derive_prediction_governance(row)
        probability_states.append(state)
        governance_reasons.append(reason)
        calibrated_up.append(up)
        calibrated_down.append(down)
        calibrated_flat.append(flat)
        scope_dicts.append(scope if scope else None)
        scope_labels.append(_summarize_calibration_scope(scope))
        artifact_hashes.append(_resolve_calibration_artifact_hash(row))
        dp_versions.append(_resolve_decision_path_contract_version(row))
        exclusion_dicts.append(exclusions if exclusions else None)
        exclusion_labels.append(_summarize_decision_path_exclusions(exclusions))

    normalized["probability_state"] = probability_states
    normalized["governance_state"] = probability_states
    normalized["governance_reason"] = governance_reasons
    normalized["calibrated_prob_up"] = calibrated_up
    normalized["calibrated_prob_down"] = calibrated_down
    normalized["calibrated_prob_flat"] = calibrated_flat
    normalized["calibration_scope"] = scope_dicts
    normalized["calibration_scope_label"] = scope_labels
    normalized["calibration_artifact_hash"] = artifact_hashes
    normalized["decision_path_contract_version"] = dp_versions
    normalized["decision_path_exclusions"] = exclusion_dicts
    normalized["decision_path_exclusions_text"] = exclusion_labels
    return normalized


def _normalize_decision_trace_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    normalized = df.copy()
    if "quality_state" not in normalized.columns and "data_quality_state" in normalized.columns:
        normalized["quality_state"] = normalized["data_quality_state"]
    if "quality_state" not in normalized.columns:
        normalized["quality_state"] = "UNKNOWN"

    governance_states: List[str] = []
    governance_reasons: List[Optional[str]] = []
    scope_dicts: List[Optional[Dict[str, Any]]] = []
    scope_labels: List[Optional[str]] = []
    artifact_hashes: List[Optional[str]] = []
    dp_versions: List[Optional[str]] = []
    exclusion_dicts: List[Optional[Dict[str, List[str]]]] = []
    exclusion_labels: List[Optional[str]] = []

    for row in normalized.to_dict(orient="records"):
        state, reason, scope, exclusions = _derive_trace_governance(row)
        governance_states.append(state)
        governance_reasons.append(reason)
        scope_dicts.append(scope if scope else None)
        scope_labels.append(_summarize_calibration_scope(scope))
        artifact_hashes.append(_resolve_calibration_artifact_hash(row))
        dp_versions.append(_resolve_decision_path_contract_version(row))
        exclusion_dicts.append(exclusions if exclusions else None)
        exclusion_labels.append(_summarize_decision_path_exclusions(exclusions))

    normalized["governance_state"] = governance_states
    normalized["governance_reason"] = governance_reasons
    normalized["calibration_scope"] = scope_dicts
    normalized["calibration_scope_label"] = scope_labels
    normalized["calibration_artifact_hash"] = artifact_hashes
    normalized["decision_path_contract_version"] = dp_versions
    normalized["decision_path_exclusions"] = exclusion_dicts
    normalized["decision_path_exclusions_text"] = exclusion_labels
    return normalized


def build_prediction_contract_frame(con: duckdb.DuckDBPyConnection, snapshot_id: str) -> pd.DataFrame:
    if not _has_table(con, "predictions"):
        return pd.DataFrame()

    select_exprs = [
        _column_expr(con, table="predictions", sql_alias="p", column=column, alias=alias)
        for column, alias in PREDICTION_REPORT_FIELDS
    ]
    query = f"""
        SELECT
          {", ".join(select_exprs)}
        FROM predictions p
        WHERE p.snapshot_id = ?
        ORDER BY COALESCE(p.horizon_seconds, p.horizon_minutes * 60, 0), p.horizon_minutes NULLS LAST
    """
    df = con.execute(query, [snapshot_id]).fetchdf()
    return _normalize_prediction_contract_frame(df)


def build_realized_prediction_contract_frame(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    if not _has_table(con, "predictions") or not _has_table(con, "snapshots"):
        return pd.DataFrame()

    pred_exprs = [
        _column_expr(con, table="predictions", sql_alias="p", column=column, alias=alias)
        for column, alias in REALIZED_REPORT_FIELDS
    ]
    snapshot_session = _column_expr(con, table="snapshots", sql_alias="s", column="session_label")
    query = f"""
        SELECT
          {", ".join(pred_exprs)},
          {snapshot_session}
        FROM predictions p
        JOIN snapshots s ON s.snapshot_id = p.snapshot_id
        WHERE COALESCE(p.outcome_realized, FALSE) = TRUE
          AND COALESCE(p.is_mock, FALSE) = FALSE
    """
    df = con.execute(query).fetchdf()
    return _normalize_prediction_contract_frame(df)


def build_calibrated_scorecard_frame(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    realized = build_realized_prediction_contract_frame(con)
    if realized.empty:
        return realized
    calibrated = realized[(realized["probability_state"] == "CALIBRATED") & (~realized["brier_score"].isna())].copy()
    if calibrated.empty:
        return calibrated
    calibrated["realized_at_et"] = pd.to_datetime(calibrated["realized_at_utc"], utc=True).dt.tz_convert(ET)
    return (
        calibrated.groupby("session_label")
        .agg(preds=("brier_score", "count"), mean_brier=("brier_score", "mean"))
        .sort_index()
        .reset_index()
    )


def build_decision_trace_frame(con: duckdb.DuckDBPyConnection, snapshot_id: str) -> pd.DataFrame:
    if not _has_table(con, "decision_traces"):
        return pd.DataFrame()

    select_exprs = [
        _column_expr(con, table="decision_traces", sql_alias="d", column=column, alias=alias)
        for column, alias in DECISION_TRACE_FIELDS
    ]
    query = f"""
        SELECT
          {", ".join(select_exprs)}
        FROM decision_traces d
        WHERE d.snapshot_id = ?
        ORDER BY d.created_at_utc DESC
        LIMIT 10
    """
    df = con.execute(query, [snapshot_id]).fetchdf()
    if not df.empty and "created_at_utc" in df.columns:
        df["created_at_utc"] = pd.to_datetime(df["created_at_utc"], utc=True).dt.tz_convert(ET)
    return _normalize_decision_trace_frame(df)


def render_prediction_contract_report(con: duckdb.DuckDBPyConnection, snapshot_id: str) -> None:
    ui = _require_streamlit()
    ui.subheader("🧠 Prediction Contract Report")
    ui.caption(
        "Governance fields shown below are storage-backed or contract-derived. Calibrated probabilities are displayed only when governance is not suppressed or unknown."
    )
    df = build_prediction_contract_frame(con, snapshot_id)
    if df.empty:
        ui.info("No persisted predictions found for the selected snapshot.")
        return

    counts = df["governance_state"].value_counts(dropna=False).to_dict()
    metric_cols = ui.columns(5)
    metric_cols[0].metric("Horizons", int(len(df)))
    metric_cols[1].metric("Calibrated", int(counts.get("CALIBRATED", 0)))
    metric_cols[2].metric("Degraded", int(counts.get("DEGRADED", 0)))
    metric_cols[3].metric("Unknown governance", int(counts.get("UNKNOWN_GOVERNANCE", 0)))
    metric_cols[4].metric("Suppressed", int(counts.get("SUPPRESSED", 0)))

    display_cols = [
        "horizon_minutes",
        "decision_state",
        "risk_gate_status",
        "quality_state",
        "confidence_state",
        "governance_state",
        "governance_reason",
        "target_version",
        "calibration_version",
        "calibration_scope_label",
        "replay_mode",
        "ood_state",
        "ood_reason",
        "decision_path_exclusions_text",
        "suppression_reason",
        "calibrated_prob_up",
        "calibrated_prob_down",
        "calibrated_prob_flat",
    ]
    ui.dataframe(df[display_cols], use_container_width=True)


def render_scorecard(con: duckdb.DuckDBPyConnection) -> None:
    ui = _require_streamlit()
    ui.subheader("📊 Calibrated Probability Scorecard (ET)")

    realized = build_realized_prediction_contract_frame(con)
    if realized.empty:
        ui.info("No realized (non-mock) predictions yet.")
        return

    excluded = int((realized["probability_state"] != "CALIBRATED").sum())
    if excluded:
        ui.caption(
            f"Excluded {excluded} realized predictions from this scorecard because their stored governance state was degraded, suppressed, unavailable, invalid, or unknown."
        )

    agg = build_calibrated_scorecard_frame(con)
    if agg.empty:
        ui.info("No realized predictions with coherent fully governed calibrated probability contracts yet.")
        return

    ui.dataframe(agg, use_container_width=True)


def render_decision_trace(con: duckdb.DuckDBPyConnection, snapshot_id: str) -> None:
    ui = _require_streamlit()
    df = build_decision_trace_frame(con, snapshot_id)
    if df.empty:
        return
    ui.subheader("🧾 Decision Trace")
    ui.dataframe(
        df[
            [
                "created_at_utc",
                "event_type",
                "decision_state",
                "risk_gate_status",
                "quality_state",
                "confidence_state",
                "governance_state",
                "governance_reason",
                "ood_state",
                "ood_reason",
                "calibration_version",
                "calibration_scope_label",
                "decision_path_exclusions_text",
                "suppression_reason",
                "replay_mode",
            ]
        ],
        use_container_width=True,
    )


def main() -> None:
    ui = _require_streamlit()
    configure_page()
    ui.title("UW Intraday Stack — Decision Support")
    ui.caption(DASHBOARD_DISCLAIMER)

    con = get_db()
    try:
        tickers = con.execute("SELECT ticker FROM dim_tickers").fetchdf()
        if tickers.empty:
            ui.info("No tickers found. Run ingestion first.")
            return

        ticker = ui.sidebar.selectbox("Ticker", tickers["ticker"].tolist())

        snap = con.execute(
            """
            SELECT
              snapshot_id, asof_ts_utc, session_label, data_quality_score,
              market_close_utc, post_end_utc, seconds_to_close, is_early_close
            FROM snapshots
            WHERE ticker=?
            ORDER BY asof_ts_utc DESC
            LIMIT 1
            """,
            [ticker],
        ).fetchone()
        if not snap:
            ui.info("No snapshots found for selected ticker.")
            return

        cols = ui.columns(4)

        ts = pd.to_datetime(snap[1], utc=True).tz_convert(ET)
        cols[0].metric("Session", snap[2], ts.strftime("%H:%M ET"))
        cols[1].metric("Data Quality", f"{float(snap[3]):.2f}")

        with cols[2]:
            close_ts = pd.to_datetime(snap[4], utc=True).tz_convert(ET) if snap[4] else None
            post_ts = pd.to_datetime(snap[5], utc=True).tz_convert(ET) if snap[5] else None
            if bool(snap[7]):
                ui.error("⚠️ EARLY CLOSE")
            if close_ts is not None:
                ui.caption(f"Close: {close_ts.strftime('%H:%M')} ET")
            if post_ts is not None:
                ui.caption(f"Post-Mkt: {post_ts.strftime('%H:%M')} ET")

        with cols[3]:
            if snap[6] is not None:
                ui.metric("Countdown", f"{int(int(snap[6]) / 60)} min")
            else:
                ui.caption("Market Closed")

        snapshot_id = str(snap[0])
        render_prediction_contract_report(con, snapshot_id)
        render_scorecard(con)
        render_decision_trace(con, snapshot_id)
    finally:
        con.close()


if __name__ == "__main__":
    main()
