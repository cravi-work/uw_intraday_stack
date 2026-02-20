from src.ingest_engine import PlannedCall, _expand

def test_path_and_query_param_expansion():
    """Asserts that both path and query parameters are dynamically injected with ticker and date."""
    call = PlannedCall(
        name="test_call",
        method="GET",
        path="/api/stock/{ticker}/test",
        path_params={"target_date": "{date}"},
        query_params={"start_date": "{date}", "limit": 50},
        is_market=False
    )
    
    path_params, query_params = _expand(call, "AAPL", "2025-10-31")
    
    # Check Path Params
    assert path_params["ticker"] == "AAPL"
    assert path_params["target_date"] == "2025-10-31"
    
    # Check Query Params
    assert query_params["start_date"] == "2025-10-31"
    assert query_params["limit"] == 50