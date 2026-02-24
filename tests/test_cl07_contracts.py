import pytest
from src.models import SessionState, DataQualityState, RiskGateStatus, SignalState, DecisionGate
from src.storage import SCHEMA_SQL
from dataclasses import fields

def test_enum_serialization_freeze():
    """
    EVIDENCE: Interface drift is caught by tests before runtime. Ensures no 
    silent renames or removals of critical frozen enum values occur.
    """
    assert "RTH" in [e.value for e in SessionState]
    assert "DEGRADED" in [e.value for e in RiskGateStatus]
    assert "NO_SIGNAL" in [e.value for e in SignalState]
    assert "PARTIAL" in [e.value for e in DataQualityState]
    
def test_prediction_schema_columns_freeze():
    """
    EVIDENCE: Prediction schema columns used by validation/reporting are frozen.
    """
    schema_upper = SCHEMA_SQL.upper()
    assert "DECISION_STATE" in schema_upper
    assert "ALIGNMENT_STATUS" in schema_upper
    assert "DECISION_WINDOW_ID" in schema_upper
    assert "CRITICAL_MISSING_COUNT" in schema_upper

def test_decision_gate_shape_freeze():
    """
    EVIDENCE: Ensures DecisionGate fields utilized by the pipeline remain rigidly in place.
    """
    f_names = [f.name for f in fields(DecisionGate)]
    assert "critical_features_missing" in f_names
    assert "validation_eligible" in f_names
    assert "risk_gate_status" in f_names