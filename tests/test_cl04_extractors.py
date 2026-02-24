import pytest
import math
from unittest.mock import MagicMock
from src.features import (
    extract_darkpool_pressure, 
    extract_litflow_pressure, 
    extract_volatility_features,
    EndpointContext
)

@pytest.fixture
def mock_ctx():
    ctx = MagicMock(spec=EndpointContext)
    ctx.freshness_state = "FRESH"
    ctx.na_reason = None
    ctx.path = "/api/mock"
    ctx.method = "GET"
    ctx.operation_id = "test"
    ctx.used_event_id = "mock-id"
    ctx.signature = "sig"
    return ctx

def test_darkpool_pressure_finite_guards(mock_ctx):
    """
    EVIDENCE: Proves darkpool metrics compute properly and strictly block NaNs/Infs.
    """
    payload = [
        {"price": 100.0, "volume": 500},
        {"price": 100.0, "volume": float('nan')},  # Should be skipped
        {"price": float('inf'), "volume": 100}     # Should be skipped
    ]
    bundle = extract_darkpool_pressure(payload, mock_ctx)
    val = bundle.features["darkpool_pressure"]
    
    assert val == 50000.0  # Only the first row should be successfully parsed
    assert math.isfinite(val)

def test_litflow_pressure_side_logic(mock_ctx):
    """
    EVIDENCE: Proves litflow evaluates correct ASK/BID sided pressure.
    """
    payload = [
        {"price": 10.0, "size": 100, "side": "ASK"}, # Bullish flow +1000
        {"price": 10.0, "size": 50, "side": "BID"}   # Bearish flow -500
    ]
    bundle = extract_litflow_pressure(payload, mock_ctx)
    val = bundle.features["litflow_pressure"]
    
    assert val == 500.0
    
def test_volatility_finite_guards(mock_ctx):
    """
    EVIDENCE: Proves volatility metrics explicitly reject NaNs to protect the pipeline.
    """
    payload = [{"iv_rank": float('nan')}]
    bundle = extract_volatility_features(payload, mock_ctx)
    
    assert bundle.features["iv_rank"] is None