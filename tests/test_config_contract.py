import copy
import textwrap

import pytest

from src.config_loader import load_yaml, summarize_effective_runtime_config
from src.ingest_engine import _validate_config


def _governed_model_section() -> dict:
    return {
        "model_name": "bounded_additive_score",
        "model_version": "1.0.0",
        "weights": {
            "spot": 1.0,
            "oi_pressure": 0.5,
            "darkpool_pressure": 0.0,
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
                        "evidence_ref": "replay://tests/calibration/fixed15",
                        "fit_sample_count": 25000,
                    },
                },
                {
                    "artifact_name": "bounded_additive_score_calibration",
                    "artifact_version": "2026.03.0.fixed60.any",
                    "target_name": "intraday_direction_3class",
                    "target_version": "2026.02.0",
                    "scope": {
                        "horizon_kind": "FIXED",
                        "horizon_minutes": 60,
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
                        "evidence_ref": "replay://tests/calibration/fixed60",
                        "fit_sample_count": 20000,
                    },
                },
                {
                    "artifact_name": "bounded_additive_score_calibration",
                    "artifact_version": "2026.03.0.toclose.any",
                    "target_name": "intraday_direction_3class",
                    "target_version": "2026.02.0",
                    "scope": {
                        "horizon_kind": "TO_CLOSE",
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
                        "evidence_ref": "replay://tests/calibration/toclose",
                        "fit_sample_count": 18000,
                    },
                },
            ],
        },
    }


@pytest.fixture
def valid_cfg():
    return {
        "ingestion": {
            "watchlist": ["AAPL"],
            "cadence_minutes": 5,
            "timezone": "America/New_York",
        },
        "storage": {"duckdb_path": "", "cycle_lock_path": "", "writer_lock_path": ""},
        "system": {"api_key_env": "UW_API_KEY"},
        "network": {"base_url": "https://api.unusualwhales.com"},
        "model": _governed_model_section(),
        "validation": {
            "invalid_after_minutes": 60,
            "tolerance_minutes": 10,
            "max_horizon_drift_minutes": 10,
            "flat_threshold_pct": 0.001,
            "fallback_max_age_minutes": 15,
            "horizons_minutes": [15, 60],
            "alignment_tolerance_sec": 900,
            "emit_to_close_horizon": True,
            "use_default_required_features": False,
            "horizon_weights_source": "model",
            "horizon_weights": {},
            "horizon_weights_overrides": {
                "15": {},
                "60": {},
                "to_close": {},
            },
            "horizon_critical_features": {
                "15": ["spot"],
                "60": ["spot"],
                "to_close": ["spot"],
            },
            "decision_path_policy": {
                "contract_version": "decision_path/v1",
                "zero_weight_is_non_decision": True,
                "require_feature_metadata": True,
                "allow_explicit_zero_weight_critical_override": True,
                "explicit_zero_weight_critical_features": {
                    "15": [],
                    "60": [],
                    "to_close": [],
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


def test_missing_required_key_fails(valid_cfg):
    del valid_cfg["validation"]["alignment_tolerance_sec"]
    with pytest.raises(KeyError, match="Missing validation.alignment_tolerance_sec"):
        _validate_config(valid_cfg)



def test_missing_to_close_criticals_fails_when_emitted(valid_cfg):
    del valid_cfg["validation"]["horizon_critical_features"]["to_close"]
    with pytest.raises(KeyError, match="Missing validation.horizon_critical_features for horizon 'to_close'"):
        _validate_config(valid_cfg)



def test_unknown_weight_key_fails_fast(valid_cfg):
    valid_cfg["model"]["weights"]["strike_flow_imbalance"] = 1.0
    with pytest.raises(ValueError, match="Unknown feature keys in config.*strike_flow_imbalance"):
        _validate_config(valid_cfg)



def test_unknown_critical_key_fails_fast(valid_cfg):
    valid_cfg["validation"]["horizon_critical_features"]["15"].append("some_unknown_feature")
    with pytest.raises(ValueError, match="Unknown feature keys in config.*some_unknown_feature"):
        _validate_config(valid_cfg)



def test_legacy_system_aliases_are_normalized(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        textwrap.dedent(
            """
            system:
              timezone: America/New_York
              base_url: https://legacy.example.test
              api_key_env: UW_API_KEY
            ingestion:
              watchlist: [AAPL]
              cadence_minutes: 5
            network: {}
            storage:
              duckdb_path: ./db.duckdb
              cycle_lock_path: ./cycle.lock
              writer_lock_path: ./writer.lock
            model:
              weights:
                spot: 1.0
            validation:
              invalid_after_minutes: 60
              tolerance_minutes: 10
              max_horizon_drift_minutes: 10
              flat_threshold_pct: 0.001
              fallback_max_age_minutes: 15
              horizons_minutes: [15]
              alignment_tolerance_sec: 900
              emit_to_close_horizon: false
              use_default_required_features: false
              horizon_weights_source: model
              horizon_weights: {}
              horizon_weights_overrides:
                "15": {}
              horizon_critical_features:
                "15": [spot]
            adapt:
              enabled: false
            """
        ),
        encoding="utf-8",
    )

    cfg = load_yaml(cfg_path).raw

    assert cfg["ingestion"]["timezone"] == "America/New_York"
    assert cfg["network"]["base_url"] == "https://legacy.example.test"



def test_conflicting_legacy_alias_fails(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        textwrap.dedent(
            """
            system:
              timezone: America/New_York
            ingestion:
              timezone: UTC
              watchlist: [AAPL]
              cadence_minutes: 5
            network: {}
            storage:
              duckdb_path: ./db.duckdb
              cycle_lock_path: ./cycle.lock
              writer_lock_path: ./writer.lock
            validation:
              invalid_after_minutes: 60
              tolerance_minutes: 10
              max_horizon_drift_minutes: 10
              flat_threshold_pct: 0.001
              fallback_max_age_minutes: 15
              horizons_minutes: [15]
              alignment_tolerance_sec: 900
              emit_to_close_horizon: false
              use_default_required_features: false
              horizon_weights_source: explicit
              horizon_weights:
                "15": {spot: 1.0}
              horizon_critical_features:
                "15": [spot]
            model:
              weights:
                spot: 1.0
            adapt:
              enabled: false
            """
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Conflicting config values for system.timezone and ingestion.timezone"):
        load_yaml(cfg_path)



def test_unused_safe_mode_key_rejected(valid_cfg):
    valid_cfg["model"]["safe_mode_enabled"] = True
    with pytest.raises(ValueError, match="Unsupported config key 'model.safe_mode_enabled'"):
        _validate_config(valid_cfg)



def test_adapt_enabled_rejected(valid_cfg):
    valid_cfg["adapt"]["enabled"] = True
    with pytest.raises(ValueError, match="adapt.enabled=true is not supported"):
        _validate_config(valid_cfg)



def test_inactive_explicit_weights_rejected_when_model_source(valid_cfg):
    valid_cfg["validation"]["horizon_weights"]["15"] = {"spot": 1.0}
    with pytest.raises(ValueError, match="validation.horizon_weights must be empty when validation.horizon_weights_source='model'"):
        _validate_config(valid_cfg)



def test_label_contract_required_fields_when_present(valid_cfg):
    del valid_cfg["validation"]["label_contract"]["threshold_policy_version"]
    with pytest.raises(KeyError, match="Missing validation.label_contract.threshold_policy_version"):
        _validate_config(valid_cfg)



def test_missing_decision_path_policy_field_rejected(valid_cfg):
    del valid_cfg["validation"]["decision_path_policy"]["require_feature_metadata"]
    with pytest.raises(KeyError, match="Missing validation.decision_path_policy.require_feature_metadata"):
        _validate_config(valid_cfg)



def test_zero_weight_critical_feature_requires_explicit_override(valid_cfg):
    valid_cfg["validation"]["horizon_critical_features"]["15"].append("darkpool_pressure")
    with pytest.raises(ValueError, match="Zero-weight critical features require explicit override"):
        _validate_config(valid_cfg)



def test_zero_weight_critical_override_is_allowed_when_explicit(valid_cfg):
    valid_cfg["validation"]["horizon_critical_features"]["15"].append("darkpool_pressure")
    valid_cfg["validation"]["decision_path_policy"]["explicit_zero_weight_critical_features"]["15"] = ["darkpool_pressure"]
    _validate_config(valid_cfg)



def test_missing_ood_assessment_policy_field_rejected(valid_cfg):
    del valid_cfg["model"]["ood_assessment_policy"]["boundary_slack"]
    with pytest.raises(KeyError, match="Missing model.ood_assessment_policy.boundary_slack"):
        _validate_config(valid_cfg)



def test_missing_ood_probability_policy_field_rejected(valid_cfg):
    del valid_cfg["model"]["ood_probability_policy"]["unknown_emit_calibrated"]
    with pytest.raises(KeyError, match="Missing model.ood_probability_policy.unknown_emit_calibrated"):
        _validate_config(valid_cfg)



def test_legacy_calibration_mapping_rejected_when_registry_governs_forward_probabilities(valid_cfg):
    valid_cfg["model"]["calibration"] = {
        "artifact_name": "legacy_calibration",
        "artifact_version": "legacy_v1",
        "bins": [0.0, 0.5, 1.0],
        "mapped": [0.05, 0.5, 0.95],
    }
    with pytest.raises(ValueError, match="model.calibration legacy mapping must be removed"):
        _validate_config(valid_cfg)



def test_missing_calibration_compatibility_rule_rejected(valid_cfg):
    del valid_cfg["model"]["calibration_registry"]["compatibility_rules"]["require_artifact_hash"]
    with pytest.raises(KeyError, match="Missing model.calibration_registry.compatibility_rules.require_artifact_hash"):
        _validate_config(valid_cfg)






def test_missing_governance_mode_defaults_for_backward_compatibility(valid_cfg):
    del valid_cfg["validation"]["governance_mode"]
    _validate_config(valid_cfg)
    summary = summarize_effective_runtime_config(valid_cfg)
    assert summary["governance_mode"] == "FORWARD_OBSERVATION"



def test_invalid_governance_mode_rejected(valid_cfg):
    valid_cfg["validation"]["governance_mode"] = "PAPER"
    with pytest.raises(ValueError, match="validation.governance_mode must be one of"):
        _validate_config(valid_cfg)



def test_missing_output_domain_policy_field_rejected(valid_cfg):
    del valid_cfg["validation"]["output_domain_policy"]["require_expected_bounds"]
    with pytest.raises(KeyError, match="Missing validation.output_domain_policy.require_expected_bounds"):
        _validate_config(valid_cfg)



def test_output_domain_contract_version_must_match_runtime(valid_cfg):
    valid_cfg["validation"]["output_domain_policy"]["required_contract_version"] = "output_domain/v2"
    with pytest.raises(ValueError, match="required_contract_version must match runtime feature contract"):
        _validate_config(valid_cfg)



def test_missing_calibration_generic_scope_policy_rejected(valid_cfg):
    del valid_cfg["model"]["calibration_registry"]["selection_policy"]["allow_generic_scope_fallback"]
    with pytest.raises(KeyError, match="Missing model.calibration_registry.selection_policy.allow_generic_scope_fallback"):
        _validate_config(valid_cfg)



def test_mismatched_required_provenance_fields_rejected(valid_cfg):
    valid_cfg["model"]["calibration_registry"]["compatibility_rules"]["required_provenance_fields"] = ["trained_from_utc"]
    with pytest.raises(ValueError, match="selection_policy.required_provenance_fields must match"):
        _validate_config(valid_cfg)



def test_missing_artifact_provenance_field_rejected(valid_cfg):
    del valid_cfg["model"]["calibration_registry"]["artifacts"][0]["provenance"]["evidence_ref"]
    with pytest.raises(KeyError, match=r"Missing model.calibration_registry.artifacts\[0\]\.evidence_ref"):
        _validate_config(valid_cfg)



def test_institutional_grade_requires_no_generic_scope_fallback(valid_cfg):
    valid_cfg["validation"]["governance_mode"] = "INSTITUTIONAL_GRADE"
    valid_cfg["model"]["calibration_registry"]["selection_policy"]["institutional_grade"] = True
    valid_cfg["model"]["calibration_registry"]["selection_policy"]["allow_generic_scope_fallback"] = True
    with pytest.raises(ValueError, match="allow_generic_scope_fallback must be false in institutional-grade mode"):
        _validate_config(valid_cfg)


def test_effective_runtime_summary_exposes_governance_contract_fields(valid_cfg):
    _validate_config(valid_cfg)
    summary = summarize_effective_runtime_config(valid_cfg)

    assert summary["timezone"] == "America/New_York"
    assert summary["base_url"] == "https://api.unusualwhales.com"
    assert summary["model_name"] == "bounded_additive_score"
    assert summary["target_name"] == "intraday_direction_3class"
    assert summary["governance_mode"] == "FORWARD_OBSERVATION"
    assert summary["decision_path_policy_version"] == "decision_path/v1"
    assert summary["zero_weight_is_non_decision"] is True
    assert summary["ood_assessment_policy_version"] == "ood_assessment/v1"
    assert summary["ood_probability_policy_version"] == "ood_probability/v1"
    assert summary["calibration_registry_version"] == "2026.03.0"
    assert summary["calibration_scope_required"] is True
    assert summary["calibration_allow_legacy_fallback"] is False
    assert summary["calibration_allow_generic_scope_fallback"] is True
    assert summary["calibration_require_provenance"] is True
    assert summary["calibration_required_provenance_fields"] == [
        "trained_from_utc",
        "trained_to_utc",
        "valid_from_utc",
        "valid_to_utc",
        "evidence_ref",
        "fit_sample_count",
    ]
    assert summary["calibration_institutional_grade"] is False
    assert summary["calibration_compatibility_rules"]["require_artifact_hash"] is True
    assert summary["calibration_compatibility_rules"]["require_provenance_fields"] is True
    assert summary["legacy_calibration_declared"] is False
    assert summary["output_domain_policy_version"] == "output_domain_policy/v1"
    assert summary["output_domain_required_contract_version"] == "output_domain/v1"
    assert summary["output_domain_requirements"]["require_expected_bounds"] is True
    assert summary["threshold_policy_version"] == "2026.02.0"
    assert summary["label_version"] == "2026.02.0"
    assert summary["adapt_enabled"] is False
    assert summary["adapt_supported"] is False
    assert "Adaptive runtime is disabled" in summary["adapt_rejection_reason"]
