from __future__ import annotations

import textwrap
from pathlib import Path

import duckdb
import pytest
from unittest.mock import patch

from src.main import main
from src.adapt import (
    ADAPT_UNSUPPORTED_REASON,
    AdaptUnsupportedError,
    adapt_config_if_needed,
    compute_recent_metrics,
    get_adapt_support_status,
    validate_adapt_config,
)
from src.config_loader import summarize_effective_runtime_config
from src.ingest_engine import _validate_config


def _valid_cfg() -> dict:
    return {
        "ingestion": {
            "watchlist": ["AAPL"],
            "cadence_minutes": 5,
            "timezone": "America/New_York",
        },
        "storage": {"duckdb_path": "", "cycle_lock_path": "", "writer_lock_path": ""},
        "system": {"api_key_env": "UW_API_KEY"},
        "network": {"base_url": "https://api.unusualwhales.com"},
        "model": {
            "model_name": "bounded_additive_score",
            "model_version": "1.0.0",
            "weights": {
                "spot": 1.0,
                "oi_pressure": 0.5,
            },
            "target_spec": {
                "target_name": "intraday_direction_3class",
                "target_version": "2026.02.0",
            },
            "ood_assessment_policy": {
                "contract_version": "ood_assessment/v1",
                "degraded_coverage_threshold": 0.85,
                "out_coverage_threshold": 0.50,
                "boundary_slack": 1.0e-06,
                "require_assessment_before_emission": True,
            },
            "ood_probability_policy": {
                "contract_version": "ood_probability/v1",
                "out_confidence_scale": 0.0,
                "out_emit_calibrated": False,
                "unknown_confidence_scale": 0.0,
                "unknown_emit_calibrated": False,
                "degraded_confidence_scale": 0.50,
                "degraded_emit_calibrated": True,
            },
            "calibration_registry": {
                "contract_version": "calibration_registry/v1",
                "registry_version": "2026.03.0",
                "default_regime": "DEFAULT",
                "selection_policy": {
                    "require_scope_match": True,
                    "allow_legacy_fallback": False,
                    "allow_generic_scope_fallback": True,
                    "require_provenance": True,
                    "required_provenance_fields": [
                        "trained_from_utc",
                        "trained_to_utc",
                        "valid_from_utc",
                        "valid_to_utc",
                        "evidence_ref",
                        "fit_sample_count",
                    ],
                    "institutional_grade": False,
                },
                "compatibility_rules": {
                    "require_target_match": True,
                    "require_horizon_match": True,
                    "require_session_match": True,
                    "require_regime_match": True,
                    "require_replay_mode_match": True,
                    "require_artifact_hash": True,
                    "require_provenance_fields": True,
                    "required_provenance_fields": [
                        "trained_from_utc",
                        "trained_to_utc",
                        "valid_from_utc",
                        "valid_to_utc",
                        "evidence_ref",
                        "fit_sample_count",
                    ],
                },
                "artifacts": [
                    {
                        "artifact_name": "bounded_additive_score_calibration",
                        "artifact_version": "2026.03.0.fixed15.any",
                        "target_name": "intraday_direction_3class",
                        "target_version": "2026.02.0",
                        "scope": {
                            "horizon_kind": "FIXED",
                            "horizon_minutes": 15,
                            "session": "ANY",
                            "regime": "DEFAULT",
                            "replay_mode": "ANY",
                        },
                        "bins": [0.0, 0.5, 1.0],
                        "mapped": [0.05, 0.5, 0.95],
                        "provenance": {
                            "trained_from_utc": "2025-01-02T14:30:00+00:00",
                            "trained_to_utc": "2025-12-31T21:00:00+00:00",
                            "valid_from_utc": "2026-01-02T14:30:00+00:00",
                            "valid_to_utc": "2026-12-31T21:00:00+00:00",
                            "evidence_ref": "replay://tests/adapt/calibration/fixed15",
                            "fit_sample_count": 25000,
                        },
                    }
                ],
            },
        },
        "validation": {
            "invalid_after_minutes": 60,
            "tolerance_minutes": 10,
            "max_horizon_drift_minutes": 10,
            "flat_threshold_pct": 0.001,
            "fallback_max_age_minutes": 15,
            "horizons_minutes": [15],
            "alignment_tolerance_sec": 900,
            "emit_to_close_horizon": False,
            "use_default_required_features": False,
            "horizon_weights_source": "model",
            "horizon_weights": {},
            "horizon_weights_overrides": {
                "15": {},
            },
            "horizon_critical_features": {
                "15": ["spot"],
            },
            "decision_path_policy": {
                "contract_version": "decision_path/v1",
                "zero_weight_is_non_decision": True,
                "require_feature_metadata": True,
                "allow_explicit_zero_weight_critical_override": True,
                "explicit_zero_weight_critical_features": {
                    "15": [],
                },
            },
            "governance_mode": "FORWARD_OBSERVATION",
            "output_domain_policy": {
                "contract_version": "output_domain_policy/v1",
                "required_contract_version": "output_domain/v1",
                "require_bounded_output_contract": True,
                "require_expected_bounds": True,
                "require_emitted_units": True,
                "require_raw_input_units": True,
                "require_output_domain": True,
                "require_bounded_output_flag": True,
                "degrade_on_missing_contract": True,
                "enforce_on_decision_eligible_only": True,
            },
            "label_contract": {
                "label_version": "2026.02.0",
                "session_boundary_rule": "TRUNCATE_TO_SESSION_CLOSE",
                "flat_threshold_policy": "ABS_RETURN_BAND",
                "threshold_policy_version": "2026.02.0",
            },
        },
        "adapt": {"enabled": False},
    }


def test_adapt_runtime_summary_is_explicitly_disabled():
    cfg = _valid_cfg()
    _validate_config(cfg)

    status = get_adapt_support_status(cfg)
    summary = summarize_effective_runtime_config(cfg)

    assert status.enabled_requested is False
    assert status.supported is False
    assert status.reason == ADAPT_UNSUPPORTED_REASON
    assert summary["adapt_enabled"] is False
    assert summary["adapt_supported"] is False
    assert summary["adapt_rejection_reason"] == ADAPT_UNSUPPORTED_REASON
    assert summary["governance_mode"] == "FORWARD_OBSERVATION"
    assert summary["output_domain_policy_version"] == "output_domain_policy/v1"
    assert summary["calibration_require_provenance"] is True



def test_adapt_enabled_rejected_cleanly_by_contract():
    cfg = _valid_cfg()
    cfg["adapt"]["enabled"] = True

    with pytest.raises(ValueError, match="adapt.enabled=true is not supported by the current runtime contract"):
        validate_adapt_config(cfg)

    with pytest.raises(ValueError, match="adapt.enabled=true is not supported by the current runtime contract"):
        _validate_config(cfg)



def test_compute_recent_metrics_uses_current_predictions_schema():
    con = duckdb.connect(database=":memory:")
    con.execute(
        """
        CREATE TABLE predictions (
            prediction_business_key TEXT,
            brier_score DOUBLE,
            log_loss DOUBLE,
            realized_at_utc TIMESTAMP,
            source_ts_max_utc TIMESTAMP,
            source_ts_min_utc TIMESTAMP
        )
        """
    )
    con.execute(
        """
        INSERT INTO predictions VALUES
            ('k1', 0.10, 0.20, TIMESTAMP '2026-03-01 15:00:00', NULL, NULL),
            ('k2', 0.30, 0.60, TIMESTAMP '2026-03-02 15:00:00', NULL, NULL),
            ('k3', NULL, 0.90, TIMESTAMP '2026-03-03 15:00:00', NULL, NULL)
        """
    )

    metrics = compute_recent_metrics(con, window=5)

    assert metrics == {"brier": 0.20, "logloss": 0.40}



def test_adapt_config_if_needed_raises_clean_disable_error_without_mutating_files(tmp_path: Path):
    cfg_path = tmp_path / "config.yaml"
    original_yaml = textwrap.dedent(
        """
        adapt:
          enabled: false
        model:
          confidence_cap: 0.55
          flat_from_data_quality_scale: 0.9
          weights:
            spot: 1.0
        """
    ).strip() + "\n"
    cfg_path.write_text(original_yaml, encoding="utf-8")

    con = duckdb.connect(database=":memory:")

    with pytest.raises(AdaptUnsupportedError, match="Adaptive runtime is disabled under the current contract"):
        adapt_config_if_needed(
            con,
            run_id="run-1",
            config_path=str(cfg_path),
            config_version_from_db=1,
            window=50,
            drift_warn={"brier_warn": 0.2, "logloss_warn": 0.8},
            drift_bad={"brier_bad": 0.3, "logloss_bad": 1.2},
            bounds={
                "weights": {"min": -1.0, "max": 1.0, "max_step": 0.05},
                "confidence_cap": {"min": 0.55, "max": 0.9, "max_step": 0.05},
                "flat_from_data_quality_scale": {"min": 0.1, "max": 0.9, "max_step": 0.1},
            },
        )

    assert cfg_path.read_text(encoding="utf-8") == original_yaml
    assert not list(tmp_path.glob("config.*.yaml"))



def test_ingest_cli_rejects_adapt_enabled_cleanly(capsys):
    cfg = _valid_cfg()
    cfg["adapt"]["enabled"] = True

    test_args = ["main.py", "ingest-once", "--config", "dummy.yaml"]
    with patch("sys.argv", test_args), patch("src.main.load_yaml") as mock_load:
        mock_load.return_value.raw = cfg

        with pytest.raises(SystemExit) as exc:
            main()

        assert exc.value.code == 1
        stderr = capsys.readouterr().err
        assert "Config Validation Error:" in stderr
        assert "adapt.enabled=true is not supported by the current runtime contract" in stderr
