import pytest
import datetime
from datetime import timezone
from src.capabilities import (
    _safe_age_seconds, 
    _validate_db_schema, 
    extract_db_truth, 
    SchemaMismatchError
)

# --- Mocks ---

class MockConSchema:
    """Mocks a duckdb connection for PRAGMA table_info queries."""
    def __init__(self, existing_columns):
        self.existing_columns = existing_columns

    def execute(self, query):
        return self

    def fetchall(self):
        # PRAGMA table_info returns tuples where index 1 is the column name
        return [(i, col) for i, col in enumerate(self.existing_columns)]


class MockConData:
    """Mocks a duckdb connection for capability truth queries."""
    def execute(self, query):
        return self

    def fetchall(self):
        now = datetime.datetime.now(timezone.utc)
        
        # INCREASED to 130 seconds to prevent microsecond execution time truncation flakes
        stale_time = now - datetime.timedelta(seconds=130) 
        
        # Tuple format: 
        # (ticker, method, path, attempt_ts, success_ts, change_ts, http_status, err_type, err_msg)
        return [
            # 1. Healthy Endpoint (AAPL)
            ("AAPL", "GET", "/api/healthy", now, now, now, 200, None, None),
            
            # 2. Failing Endpoint (SPY)
            ("SPY", "GET", "/api/failing", now, stale_time, stale_time, 500, "HttpStatusError", "HTTP 500"),
            
            # 3. Market Endpoint (Stale payload, but healthy fetch)
            ("__MARKET__", "GET", "/api/market", now, now, stale_time, 200, None, None)
        ]

# --- Tests ---

def test_safe_age_seconds():
    now = datetime.datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    past = now - datetime.timedelta(seconds=45)
    
    assert _safe_age_seconds(past, now) == 45
    assert _safe_age_seconds(None, now) is None

def test_validate_db_schema_passes():
    """Asserts schema validation succeeds when all required columns are present."""
    valid_cols = [
        "ticker", "endpoint_id", "last_success_event_id", "last_success_ts_utc",
        "last_payload_hash", "last_change_ts_utc", "last_change_event_id",
        "last_attempt_event_id", "last_attempt_ts_utc", "last_attempt_http_status",
        "last_attempt_error_type", "last_attempt_error_msg"
    ]
    con = MockConSchema(valid_cols)
    _validate_db_schema(con)  # Should not raise any exceptions

def test_validate_db_schema_fails_on_missing():
    """Asserts schema validation fails fast if attempt columns are missing."""
    invalid_cols = [
        "ticker", "endpoint_id", "last_success_event_id", "last_success_ts_utc"
    ]
    con = MockConSchema(invalid_cols)
    
    with pytest.raises(SchemaMismatchError) as exc_info:
        _validate_db_schema(con)
        
    assert "last_attempt_event_id" in str(exc_info.value)

def test_extract_db_truth():
    """Asserts db truth properly separates attempt, success, and payload-change metrics."""
    con = MockConData()
    truth = extract_db_truth(con)
    
    # Check AAPL (Healthy)
    assert "AAPL" in truth
    aapl_ep = truth["AAPL"]["endpoints"]["GET /api/healthy"]
    assert aapl_ep["health_status"] == "HEALTHY"
    assert aapl_ep["attempt_age_s"] == 0
    assert aapl_ep["success_age_s"] == 0
    
    # Check SPY (Failing)
    assert "SPY" in truth
    spy_ep = truth["SPY"]["endpoints"]["GET /api/failing"]
    assert spy_ep["health_status"] == "ERROR"
    assert spy_ep["error_type"] == "HttpStatusError"
    
    # Check Market Context
    assert "__MARKET__" in truth
    mkt_ep = truth["__MARKET__"]["endpoints"]["GET /api/market"]
    assert mkt_ep["health_status"] == "HEALTHY"
    assert mkt_ep["attempt_age_s"] == 0
    assert mkt_ep["payload_change_age_s"] >= 120