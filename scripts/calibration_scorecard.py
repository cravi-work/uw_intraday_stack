from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import duckdb
import pandas as pd

from src.dashboard_app import build_realized_prediction_contract_frame

DEFAULT_PREFIX = "calibration_scorecard"
BIN_EDGES: Tuple[float, ...] = tuple(i / 10.0 for i in range(11))


def _clean_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    return text or None


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _round_float(value: Any, digits: int = 8) -> Optional[float]:
    num = _coerce_float(value)
    if num is None:
        return None
    return round(num, digits)


def _stable_unique(values: Iterable[Any]) -> List[str]:
    out: List[str] = []
    for value in values:
        text = _clean_text(value)
        if text is not None:
            out.append(text)
    return sorted(set(out))


def _scope_field(scope: Any, key: str) -> Optional[str]:
    if isinstance(scope, Mapping):
        return _clean_text(scope.get(key))
    if isinstance(scope, str) and scope.strip().startswith("{"):
        try:
            parsed = json.loads(scope)
        except Exception:
            return None
        if isinstance(parsed, Mapping):
            return _clean_text(parsed.get(key))
    return None


def _classify_population(state: Any) -> str:
    normalized = (_clean_text(state) or "UNKNOWN").upper()
    if normalized in {"CALIBRATED", "DEGRADED"}:
        return "ACCEPTED"
    if normalized == "SUPPRESSED":
        return "SUPPRESSED"
    return "OTHER"


def _predicted_class(row: Mapping[str, Any]) -> Optional[str]:
    pairs = [
        ("UP", _coerce_float(row.get("calibrated_prob_up"))),
        ("DOWN", _coerce_float(row.get("calibrated_prob_down"))),
        ("FLAT", _coerce_float(row.get("calibrated_prob_flat"))),
    ]
    valid = [(label, value) for label, value in pairs if value is not None]
    if not valid:
        return None
    priority = {"UP": 2, "DOWN": 1, "FLAT": 0}
    valid.sort(key=lambda item: (item[1], priority[item[0]]), reverse=True)
    return valid[0][0]


def _predicted_confidence(row: Mapping[str, Any]) -> Optional[float]:
    values = [
        _coerce_float(row.get("calibrated_prob_up")),
        _coerce_float(row.get("calibrated_prob_down")),
        _coerce_float(row.get("calibrated_prob_flat")),
    ]
    valid = [v for v in values if v is not None]
    return max(valid) if valid else None


def _probability_bin(value: Any) -> Optional[str]:
    num = _coerce_float(value)
    if num is None:
        return None
    if num < 0.0 or num > 1.0:
        return "OUT_OF_RANGE"
    for lower, upper in zip(BIN_EDGES[:-1], BIN_EDGES[1:]):
        if num < upper or math.isclose(num, upper):
            return f"{lower:.1f}-{upper:.1f}"
    return "0.9-1.0"


def _bool_from_value(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def _prepare_rows(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    if prepared.empty:
        prepared["decision_population"] = pd.Series(dtype="object")
        prepared["predicted_label"] = pd.Series(dtype="object")
        prepared["predicted_confidence"] = pd.Series(dtype="float")
        prepared["probability_bin"] = pd.Series(dtype="object")
        prepared["empirical_correct"] = pd.Series(dtype="boolean")
        return prepared

    prepared["decision_population"] = prepared["probability_state"].map(_classify_population)
    prepared["predicted_label"] = [
        _predicted_class(row) for row in prepared.to_dict(orient="records")
    ]
    prepared["predicted_confidence"] = [
        _predicted_confidence(row) for row in prepared.to_dict(orient="records")
    ]
    prepared["probability_bin"] = prepared["predicted_confidence"].map(_probability_bin)
    prepared["empirical_correct"] = [
        _bool_from_value(row.get("is_correct"))
        if row.get("is_correct") is not None
        else (
            None
            if _predicted_class(row) is None or _clean_text(row.get("outcome_label")) is None
            else _predicted_class(row) == _clean_text(row.get("outcome_label"))
        )
        for row in prepared.to_dict(orient="records")
    ]
    prepared["calibration_scope_session"] = [
        _scope_field(scope, "session") for scope in prepared.get("calibration_scope", pd.Series([None] * len(prepared)))
    ]
    prepared["calibration_scope_regime"] = [
        _scope_field(scope, "regime") for scope in prepared.get("calibration_scope", pd.Series([None] * len(prepared)))
    ]
    prepared["calibration_scope_horizon"] = [
        _scope_field(scope, "horizon_minutes") or _scope_field(scope, "horizon_kind")
        for scope in prepared.get("calibration_scope", pd.Series([None] * len(prepared)))
    ]
    prepared["calibration_scope_replay_mode"] = [
        _scope_field(scope, "replay_mode") for scope in prepared.get("calibration_scope", pd.Series([None] * len(prepared)))
    ]
    return prepared


def _dataset_fingerprint(df: pd.DataFrame) -> str:
    if df.empty:
        return hashlib.sha256(b"empty").hexdigest()
    subset = df.copy()
    cols = [
        c
        for c in [
            "snapshot_id",
            "horizon_minutes",
            "session_label",
            "replay_mode",
            "probability_state",
            "calibration_version",
            "calibration_artifact_hash",
            "calibration_evidence_ref",
            "outcome_label",
            "brier_score",
            "log_loss",
        ]
        if c in subset.columns
    ]
    subset = subset[cols].copy()
    for col in subset.columns:
        subset[col] = subset[col].map(lambda v: None if (isinstance(v, float) and math.isnan(v)) else v)
    payload = json.dumps(subset.sort_values(cols).to_dict(orient="records"), sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _summarize_overall(df: pd.DataFrame) -> Dict[str, Any]:
    population_counts = {k: int(v) for k, v in df["decision_population"].value_counts(dropna=False).to_dict().items()}
    governance_counts = {k: int(v) for k, v in df["probability_state"].value_counts(dropna=False).to_dict().items()}
    metrics_available = int(df["brier_score"].notna().sum()) if "brier_score" in df.columns else 0
    accepted_mask = df["decision_population"] == "ACCEPTED"
    accepted_metrics = df.loc[accepted_mask & df["brier_score"].notna()].copy()
    suppressed_metrics = df.loc[(df["decision_population"] == "SUPPRESSED") & df["brier_score"].notna()].copy()

    summary = {
        "prediction_count": int(len(df)),
        "metrics_available_count": metrics_available,
        "population_counts": population_counts,
        "governance_state_counts": governance_counts,
        "distinct_calibration_versions": _stable_unique(df.get("calibration_version", [])),
        "distinct_calibration_artifact_hashes": _stable_unique(df.get("calibration_artifact_hash", [])),
        "distinct_calibration_evidence_refs": _stable_unique(df.get("calibration_evidence_ref", [])),
        "dataset_fingerprint": _dataset_fingerprint(df),
        "accepted_metrics": {
            "count": int(len(accepted_metrics)),
            "mean_brier": _round_float(accepted_metrics["brier_score"].mean()) if not accepted_metrics.empty else None,
            "mean_log_loss": _round_float(accepted_metrics["log_loss"].mean()) if not accepted_metrics.empty else None,
            "accuracy_rate": _round_float(accepted_metrics["empirical_correct"].astype(float).mean()) if not accepted_metrics.empty and accepted_metrics["empirical_correct"].notna().any() else None,
        },
        "suppressed_metrics": {
            "count": int(len(suppressed_metrics)),
            "mean_brier": _round_float(suppressed_metrics["brier_score"].mean()) if not suppressed_metrics.empty else None,
            "mean_log_loss": _round_float(suppressed_metrics["log_loss"].mean()) if not suppressed_metrics.empty else None,
        },
    }
    return summary



def _aggregate_segment(group: pd.DataFrame) -> pd.Series:
    correct = group["empirical_correct"].dropna()
    return pd.Series(
        {
            "prediction_count": int(len(group)),
            "metrics_available_count": int(group["brier_score"].notna().sum()),
            "probability_available_count": int(group["predicted_confidence"].notna().sum()),
            "mean_brier": _round_float(group["brier_score"].mean()),
            "mean_log_loss": _round_float(group["log_loss"].mean()),
            "accuracy_rate": _round_float(correct.astype(float).mean()) if not correct.empty else None,
            "mean_predicted_confidence": _round_float(group["predicted_confidence"].mean()),
            "calibration_versions": ";".join(_stable_unique(group["calibration_version"])),
            "calibration_artifact_hashes": ";".join(_stable_unique(group["calibration_artifact_hash"])),
            "calibration_evidence_refs": ";".join(_stable_unique(group["calibration_evidence_ref"])),
            "governance_states": ";".join(_stable_unique(group["probability_state"])),
        }
    )



def _grouped_aggregate(df: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for keys, group in df.groupby(list(group_cols), dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: key for col, key in zip(group_cols, keys)}
        row.update(_aggregate_segment(group).to_dict())
        rows.append(row)
    return pd.DataFrame(rows)


def _build_segment_frame(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "horizon_minutes",
        "session_label",
        "replay_mode",
        "decision_population",
    ]
    if df.empty:
        return pd.DataFrame(columns=list(group_cols) + [
            "prediction_count",
            "metrics_available_count",
            "probability_available_count",
            "mean_brier",
            "mean_log_loss",
            "accuracy_rate",
            "mean_predicted_confidence",
            "calibration_versions",
            "calibration_artifact_hashes",
            "calibration_evidence_refs",
            "governance_states",
        ])
    segment = _grouped_aggregate(df, group_cols)
    return segment.sort_values(group_cols, kind="stable").reset_index(drop=True)



def _build_reliability_frame(df: pd.DataFrame) -> pd.DataFrame:
    rows = df[df["predicted_confidence"].notna()].copy()
    columns = [
        "horizon_minutes",
        "session_label",
        "replay_mode",
        "decision_population",
        "probability_bin",
        "prediction_count",
        "mean_predicted_confidence",
        "empirical_accuracy",
        "mean_brier",
        "mean_log_loss",
        "calibration_artifact_hashes",
        "calibration_versions",
    ]
    if rows.empty:
        return pd.DataFrame(columns=columns)
    group_cols = [
        "horizon_minutes",
        "session_label",
        "replay_mode",
        "decision_population",
        "probability_bin",
    ]
    records: List[Dict[str, Any]] = []
    for keys, group in rows.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        correct = group["empirical_correct"].dropna()
        record = {col: key for col, key in zip(group_cols, keys)}
        record.update(
            {
                "prediction_count": int(len(group)),
                "mean_predicted_confidence": _round_float(group["predicted_confidence"].mean()),
                "empirical_accuracy": _round_float(correct.astype(float).mean()) if not correct.empty else None,
                "mean_brier": _round_float(group["brier_score"].mean()),
                "mean_log_loss": _round_float(group["log_loss"].mean()),
                "calibration_artifact_hashes": ";".join(_stable_unique(group["calibration_artifact_hash"])),
                "calibration_versions": ";".join(_stable_unique(group["calibration_version"])),
            }
        )
        records.append(record)
    return pd.DataFrame(records).sort_values(group_cols, kind="stable").reset_index(drop=True)



def _build_population_comparison(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["decision_population"] + [
            "prediction_count",
            "metrics_available_count",
            "probability_available_count",
            "mean_brier",
            "mean_log_loss",
            "accuracy_rate",
            "mean_predicted_confidence",
            "calibration_versions",
            "calibration_artifact_hashes",
            "calibration_evidence_refs",
            "governance_states",
        ])
    comparison = _grouped_aggregate(df, ["decision_population"])
    return comparison.sort_values(["decision_population"], kind="stable").reset_index(drop=True)



def _build_artifact_inventory(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "calibration_version",
        "calibration_artifact_hash",
        "calibration_evidence_ref",
        "calibration_scope_session",
        "calibration_scope_regime",
        "calibration_scope_horizon",
        "calibration_scope_replay_mode",
    ]
    if df.empty:
        return pd.DataFrame(columns=group_cols + [
            "prediction_count",
            "metrics_available_count",
            "replay_modes_seen",
            "governance_states",
            "sessions_seen",
            "horizons_seen",
        ])
    records: List[Dict[str, Any]] = []
    for keys, group in df.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        record = {col: key for col, key in zip(group_cols, keys)}
        record.update(
            {
                "prediction_count": int(len(group)),
                "metrics_available_count": int(group["brier_score"].notna().sum()),
                "replay_modes_seen": ";".join(_stable_unique(group["replay_mode"])),
                "governance_states": ";".join(_stable_unique(group["probability_state"])),
                "sessions_seen": ";".join(_stable_unique(group["session_label"])),
                "horizons_seen": ";".join(_stable_unique(group["horizon_minutes"])),
            }
        )
        records.append(record)
    return pd.DataFrame(records).sort_values(group_cols, kind="stable").reset_index(drop=True)



def _json_ready(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return [_json_ready(record) for record in value.to_dict(orient="records")]
    if isinstance(value, Mapping):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_ready(v) for v in value]
    if isinstance(value, tuple):
        return [_json_ready(v) for v in value]
    if isinstance(value, float):
        if math.isnan(value):
            return None
        return round(value, 8)
    if pd.isna(value):  # type: ignore[arg-type]
        return None
    return value



def _df_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_(empty)_"
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows: List[str] = [header, sep]
    for _, row in df.iterrows():
        cells = []
        for col in cols:
            value = row[col]
            if pd.isna(value):
                cells.append("")
            else:
                cells.append(str(value))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)



def build_scorecard_package(con: duckdb.DuckDBPyConnection) -> Dict[str, Any]:
    realized = build_realized_prediction_contract_frame(con)
    prepared = _prepare_rows(realized)
    overall = _summarize_overall(prepared)
    segments = _build_segment_frame(prepared)
    reliability = _build_reliability_frame(prepared)
    population = _build_population_comparison(prepared)
    artifacts = _build_artifact_inventory(prepared)
    return {
        "overall_summary": overall,
        "realized_predictions": prepared,
        "segment_summary": segments,
        "reliability_summary": reliability,
        "accepted_vs_suppressed": population,
        "artifact_inventory": artifacts,
    }



def _write_outputs(package: Mapping[str, Any], output_dir: Path, prefix: str) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    overall = package["overall_summary"]
    segments = package["segment_summary"]
    reliability = package["reliability_summary"]
    population = package["accepted_vs_suppressed"]
    artifacts = package["artifact_inventory"]

    json_path = output_dir / f"{prefix}.json"
    segments_path = output_dir / f"{prefix}_segments.csv"
    reliability_path = output_dir / f"{prefix}_reliability.csv"
    population_path = output_dir / f"{prefix}_accepted_vs_suppressed.csv"
    artifacts_path = output_dir / f"{prefix}_artifact_inventory.csv"
    markdown_path = output_dir / f"{prefix}.md"

    json_payload = {
        "overall_summary": _json_ready(overall),
        "segment_summary": _json_ready(segments),
        "reliability_summary": _json_ready(reliability),
        "accepted_vs_suppressed": _json_ready(population),
        "artifact_inventory": _json_ready(artifacts),
    }
    json_path.write_text(json.dumps(json_payload, indent=2, sort_keys=True), encoding="utf-8")
    segments.to_csv(segments_path, index=False)
    reliability.to_csv(reliability_path, index=False)
    population.to_csv(population_path, index=False)
    artifacts.to_csv(artifacts_path, index=False)

    md_lines = [
        "# Calibration Scorecard",
        "",
        "## Overall Summary",
        "",
        f"- Predictions: {overall['prediction_count']}",
        f"- Metrics available: {overall['metrics_available_count']}",
        f"- Dataset fingerprint: `{overall['dataset_fingerprint']}`",
        f"- Calibration versions: {', '.join(overall['distinct_calibration_versions']) or '(none)'}",
        f"- Calibration artifact hashes: {', '.join(overall['distinct_calibration_artifact_hashes']) or '(none)'}",
        f"- Calibration evidence refs: {', '.join(overall['distinct_calibration_evidence_refs']) or '(none)'}",
        "",
        "## Population Counts",
        "",
    ]
    for key, value in sorted(overall["population_counts"].items()):
        md_lines.append(f"- {key}: {value}")
    md_lines.extend([
        "",
        "## Accepted vs Suppressed",
        "",
        _df_to_markdown(population),
        "",
        "## Segment Summary",
        "",
        _df_to_markdown(segments),
        "",
        "## Reliability Summary",
        "",
        _df_to_markdown(reliability),
        "",
        "## Artifact Inventory",
        "",
        _df_to_markdown(artifacts),
        "",
    ])
    markdown_path.write_text("\n".join(md_lines), encoding="utf-8")

    return {
        "json": json_path,
        "segments_csv": segments_path,
        "reliability_csv": reliability_path,
        "accepted_vs_suppressed_csv": population_path,
        "artifact_inventory_csv": artifacts_path,
        "markdown": markdown_path,
    }



def generate_calibration_scorecard(db_path: str | Path, output_dir: str | Path, *, prefix: str = DEFAULT_PREFIX) -> Dict[str, Path]:
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        package = build_scorecard_package(con)
    finally:
        con.close()
    return _write_outputs(package, Path(output_dir), prefix)



def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate replay-native calibration evidence package from stored predictions.")
    parser.add_argument("--db", required=True, help="Path to DuckDB database")
    parser.add_argument("--out-dir", required=True, help="Directory to write evidence files")
    parser.add_argument("--prefix", default=DEFAULT_PREFIX, help="Output file prefix")
    return parser.parse_args(argv)



def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    outputs = generate_calibration_scorecard(args.db, args.out_dir, prefix=args.prefix)
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
