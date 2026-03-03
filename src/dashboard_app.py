from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
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
    "Probability columns are shown only when the stored probability contract is calibrated, coherent, and unsuppressed."
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
    ("suppression_reason", None),
    ("prob_up", None),
    ("prob_down", None),
    ("prob_flat", None),
    ("confidence", None),
    ("probability_contract_json", None),
)

DECISION_TRACE_FIELDS: Sequence[PredictionField] = (
    ("event_type", None),
    ("decision_state", None),
    ("risk_gate_status", None),
    ("data_quality_state", "quality_state"),
    ("confidence_state", None),
    ("suppression_reason", None),
    ("ood_state", None),
    ("replay_mode", None),
    ("model_version", None),
    ("target_version", None),
    ("calibration_version", None),
    ("created_at_utc", None),
)

REALIZED_REPORT_FIELDS: Sequence[PredictionField] = (
    ("realized_at_utc", None),
    ("brier_score", None),
    ("data_quality_state", "quality_state"),
    ("target_version", None),
    ("calibration_version", None),
    ("replay_mode", None),
    ("ood_state", None),
    ("suppression_reason", None),
    ("prob_up", None),
    ("prob_down", None),
    ("prob_flat", None),
    ("probability_contract_json", None),
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


def _clean_optional_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    text = str(value).strip()
    return text if text else None


def _vector_component(vector: Any, key: str) -> Optional[float]:
    if not isinstance(vector, Mapping):
        return None
    value = vector.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _derive_probability_state(row: Mapping[str, Any]) -> Tuple[str, Optional[float], Optional[float], Optional[float]]:
    contract = _json_dict(row.get("probability_contract_json"))
    suppression_reason = _clean_optional_text(row.get("suppression_reason") or contract.get("suppression_reason"))
    ood_state = _clean_optional_text(row.get("ood_state") or contract.get("ood_state"))
    target_version = _clean_optional_text(row.get("target_version"))
    calibration_version = _clean_optional_text(row.get("calibration_version"))
    coherent = bool(contract.get("is_coherent"))

    calibrated_vector = contract.get("calibrated_probability_vector")
    display_up = row.get("prob_up")
    display_down = row.get("prob_down")
    display_flat = row.get("prob_flat")

    if suppression_reason is not None or ood_state == "OUT_OF_DISTRIBUTION":
        return "SUPPRESSED", None, None, None
    if calibrated_vector is None:
        return "UNAVAILABLE", None, None, None
    if not coherent:
        return "INVALID", None, None, None
    if target_version is None or calibration_version is None:
        return "UNVERIFIED", None, None, None

    def _coerce_prob(candidate: Any, key: str) -> Optional[float]:
        if candidate is not None and not (isinstance(candidate, float) and pd.isna(candidate)):
            try:
                return float(candidate)
            except Exception:
                pass
        return _vector_component(calibrated_vector, key)

    return (
        "CALIBRATED",
        _coerce_prob(display_up, "UP"),
        _coerce_prob(display_down, "DOWN"),
        _coerce_prob(display_flat, "FLAT"),
    )


def _normalize_prediction_contract_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    normalized = df.copy()
    if "quality_state" not in normalized.columns and "data_quality_state" in normalized.columns:
        normalized["quality_state"] = normalized["data_quality_state"]
    if "quality_state" not in normalized.columns:
        normalized["quality_state"] = "UNKNOWN"

    probability_states: List[str] = []
    calibrated_up: List[Optional[float]] = []
    calibrated_down: List[Optional[float]] = []
    calibrated_flat: List[Optional[float]] = []

    for row in normalized.to_dict(orient="records"):
        state, up, down, flat = _derive_probability_state(row)
        probability_states.append(state)
        calibrated_up.append(up)
        calibrated_down.append(down)
        calibrated_flat.append(flat)

    normalized["probability_state"] = probability_states
    normalized["calibrated_prob_up"] = calibrated_up
    normalized["calibrated_prob_down"] = calibrated_down
    normalized["calibrated_prob_flat"] = calibrated_flat
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
    return df


def render_prediction_contract_report(con: duckdb.DuckDBPyConnection, snapshot_id: str) -> None:
    ui = _require_streamlit()
    ui.subheader("🧠 Prediction Contract Report")
    ui.caption(
        "Contract fields shown below are storage-backed. Calibrated probabilities are displayed only when the stored contract is coherent and unsuppressed."
    )
    df = build_prediction_contract_frame(con, snapshot_id)
    if df.empty:
        ui.info("No persisted predictions found for the selected snapshot.")
        return

    counts = df["probability_state"].value_counts(dropna=False).to_dict()
    metric_cols = ui.columns(5)
    metric_cols[0].metric("Horizons", int(len(df)))
    metric_cols[1].metric("Calibrated", int(counts.get("CALIBRATED", 0)))
    metric_cols[2].metric("Suppressed", int(counts.get("SUPPRESSED", 0)))
    metric_cols[3].metric("OOD", int((df["ood_state"] == "OUT_OF_DISTRIBUTION").sum()))
    metric_cols[4].metric(
        "Degraded quality",
        int(df["quality_state"].astype(str).str.upper().isin({"PARTIAL", "DEGRADED", "STALE"}).sum()),
    )

    display_cols = [
        "horizon_minutes",
        "decision_state",
        "risk_gate_status",
        "quality_state",
        "confidence_state",
        "target_version",
        "calibration_version",
        "replay_mode",
        "ood_state",
        "suppression_reason",
        "probability_state",
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
            f"Excluded {excluded} realized predictions from this scorecard because their stored probability contracts were suppressed, unavailable, invalid, or unverified."
        )

    agg = build_calibrated_scorecard_frame(con)
    if agg.empty:
        ui.info("No realized predictions with coherent calibrated probability contracts yet.")
        return

    ui.dataframe(agg, use_container_width=True)


def render_decision_trace(con: duckdb.DuckDBPyConnection, snapshot_id: str) -> None:
    ui = _require_streamlit()
    df = build_decision_trace_frame(con, snapshot_id)
    if df.empty:
        return
    ui.subheader("🧾 Decision Trace")
    ui.dataframe(df, use_container_width=True)


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
