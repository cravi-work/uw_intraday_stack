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
    ctx.endpoint_id = 1
    ctx.stale_age_min = 0
    ctx.effective_ts_utc = None
    return ctx

def test_darkpool_pressure_finite_guards(mock_ctx):
    payload = [
        {"price": 100.0, "volume": 500},
        {"price": 100.0, "volume": float('nan')},  
        {"price": float('inf'), "volume": 100}     
    ]
    bundle = extract_darkpool_pressure(payload, mock_ctx)
    val = bundle.features["darkpool_pressure"]
    
    assert val == 50000.0  
    assert math.isfinite(val)

def test_litflow_pressure_side_logic(mock_ctx):
    payload = [
        {"price": 10.0, "size": 100, "side": "ASK"}, 
        {"price": 10.0, "size": 50, "side": "BID"}   
    ]
    bundle = extract_litflow_pressure(payload, mock_ctx)
    val = bundle.features["litflow_pressure"]
    
    assert val == 500.0
    
def test_volatility_finite_guards(mock_ctx):
    payload = [{"iv_rank": float('nan')}]
    bundle = extract_volatility_features(payload, mock_ctx)
    
    assert bundle.features["iv_rank"] is None