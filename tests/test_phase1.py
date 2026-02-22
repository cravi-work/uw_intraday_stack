import pytest
import datetime as dt
from zoneinfo import ZoneInfo
from src.scheduler import get_market_hours, MarketHours

ET = ZoneInfo("America/New_York")

# [Review Item 4] Config matches scheduler expectations
MOCK_CFG = {
    "premarket_start_et": "04:00",
    "regular_start_et": "09:30",
    "regular_end_et": "16:00",
    "afterhours_end_et": "20:00",
    "ingest_start_et": "04:00",
    "ingest_end_et": "18:00" 
}

def test_safe_closed_on_missing_calendar(monkeypatch):
    """[Review Item 2C] Verify system defaults to Fail Fast (or Safe Closed if forced)."""
    import src.scheduler
    monkeypatch.setattr(src.scheduler, "HAS_CALENDAR", False)
    
    # Use a hardcoded weekday (e.g., Wednesday) so the test doesn't fail if run on a weekend
    test_date = dt.date(2023, 11, 15)
    
    # Default behavior: Raise Error
    with pytest.raises(RuntimeError):
        get_market_hours(test_date, MOCK_CFG)
    
    # Configured Fallback: Return Degraded
    cfg_allow = MOCK_CFG.copy()
    cfg_allow["allow_degraded_calendar"] = True
    hours = get_market_hours(test_date, cfg_allow)
    assert hours.reason == "DEGRADED_NO_CALENDAR"

def test_to_close_targets_correct_boundary():
    """Verify seconds_to_close targets market_close in REG, post_end in AFT."""
    date = dt.date(2023, 11, 24) # Black Friday (Close 13:00)
    hours = get_market_hours(date, MOCK_CFG)
    
    # 12:00 ET (In REG) -> Targets Market Close (13:00)
    now_reg = dt.datetime(2023, 11, 24, 12, 0, tzinfo=ET)
    assert hours.seconds_to_close(now_reg) == 3600
    
    # 14:00 ET (In AFT) -> Targets Post End (17:00)
    now_aft = dt.datetime(2023, 11, 24, 14, 0, tzinfo=ET)
    # 14:00 to 17:00 = 3 hours = 10800 sec
    assert hours.seconds_to_close(now_aft) == 10800

def test_ingest_cap_logic():
    date = dt.date(2023, 11, 14)
    hours = get_market_hours(date, MOCK_CFG)
    # Normal day: Post end 20:00, Ingest Cap 18:00 -> Ingest End 18:00
    assert hours.ingest_end_et.time() == dt.time(18, 0)

def test_sanity_checks():
    bad_cfg = MOCK_CFG.copy()
    bad_cfg["ingest_start_et"] = "19:00" # After ingest end
    test_date = dt.date(2023, 11, 15)
    
    # [Review Item 4] Fix: Match the actual error message substring or type
    with pytest.raises(ValueError, match="ingest_start must be before ingest_end"):
        get_market_hours(test_date, bad_cfg)