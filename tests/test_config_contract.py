import textwrap

import pytest

from src.config_loader import load_yaml, summarize_effective_runtime_config
from src.ingest_engine import _validate_config


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
            "calibration": {
                "artifact_name": "bounded_additive_score_calibration",
                "artifact_version": "2026.02.0",
                "bins": [0.0, 0.5, 1.0],
                "mapped": [0.05, 0.5, 0.95],
            },
        },
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


def test_effective_runtime_summary_exposes_consumed_contract_fields(valid_cfg):
    _validate_config(valid_cfg)
    summary = summarize_effective_runtime_config(valid_cfg)

    assert summary["timezone"] == "America/New_York"
    assert summary["base_url"] == "https://api.unusualwhales.com"
    assert summary["model_name"] == "bounded_additive_score"
    assert summary["target_name"] == "intraday_direction_3class"
    assert summary["calibration_artifact"] == "bounded_additive_score_calibration"
    assert summary["threshold_policy_version"] == "2026.02.0"
    assert summary["label_version"] == "2026.02.0"
    assert summary["adapt_enabled"] is False
    assert summary["adapt_supported"] is False
    assert "Adaptive runtime is disabled" in summary["adapt_rejection_reason"]
