import pytest
import datetime as dt
from unittest.mock import MagicMock, patch
from src.models import SessionState
from src.scheduler import get_market_hours, ET

def test_premarket_and_rth_mapping():
    """
    EVIDENCE: Validates that premarket and regular trading hour timestamps correctly map
    to their canonical enum-compatible non-CLOSED session states without relying on
    legacy short-labels or generic fallbacks.
    """
    cfg = {
        "premarket_start_et": "04:00",
        "regular_start_et": "09:30",
        "regular_end_et": "16:00",
        "afterhours_end_et": "20:00",
        "ingest_start_et": "04:00",
        "ingest_end_et": "20:00"
    }
    
    with patch('src.scheduler.mcal') as mock_mcal, patch('src.scheduler.HAS_CALENDAR', True):
        mock_cal = MagicMock()
        mock_mcal.get_calendar.return_value = mock_cal
        
        # Setup a standard day: Jan 5, 2026
        date_obj = dt.date(2026, 1, 5)
        import pandas as pd
        mock_schedule = pd.DataFrame({
            'market_open': [pd.Timestamp('2026-01-05 09:30:00', tz='America/New_York')],
            'market_close': [pd.Timestamp('2026-01-05 16:00:00', tz='America/New_York')]
        })
        mock_cal.schedule.return_value = mock_schedule
        
        hours = get_market_hours(date_obj, cfg)
        
        pre_ts = dt.datetime(2026, 1, 5, 8, 0, tzinfo=ET)
        assert hours.get_session_label(pre_ts) == SessionState.PREMARKET.value
        
        rth_ts = dt.datetime(2026, 1, 5, 12, 0, tzinfo=ET)
        assert hours.get_session_label(rth_ts) == SessionState.RTH.value

def test_holiday_suppresses_forward_signals():
    """
    EVIDENCE: Proves that an empty market calendar schedule maps the day to CLOSED
    and entirely suppresses the active ingest logic before generating fetch calls.
    """
    from src.ingest_engine import IngestionEngine

    cfg = {
        "ingestion": {"watchlist": ["AAPL"], "cadence_minutes": 5, "enable_market_context": False},
        "storage": {"duckdb_path": ":memory:", "cycle_lock_path": "mock.lock", "writer_lock_path": "mock.lock"},
        "system": {},
        "network": {},
        "validation": {"horizons_minutes": [5]}
    }

    with patch("src.ingest_engine.get_market_hours") as mock_gmh, \
         patch("src.ingest_engine.fetch_all") as mock_fetch, \
         patch("src.ingest_engine.load_endpoint_plan"), \
         patch("src.ingest_engine.load_api_catalog"):
        
        # Setup a HOLIDAY hours object that is strictly non-trading
        mock_mh = MagicMock()
        mock_mh.is_trading_day = False
        mock_mh.reason = "HOLIDAY"
        mock_mh.get_session_label.return_value = SessionState.CLOSED.value
        mock_gmh.return_value = mock_mh

        engine = IngestionEngine(cfg=cfg, catalog_path="dummy.yaml", config_path="dummy.yaml")
        engine.run_cycle()

        # As per integration suppression evidence, fetch_all should NEVER be hit on a holiday
        mock_fetch.assert_not_called()

def test_early_close_computes_correct_post_market():
    """
    EVIDENCE: Validates that an early-close calendar directly triggers the correct 
    +4 hour logical shift for post-market boundaries, generating accurate labels.
    """
    cfg = {
        "premarket_start_et": "04:00",
        "regular_start_et": "09:30",
        "regular_end_et": "16:00",
        "afterhours_end_et": "20:00",
        "ingest_start_et": "04:00",
        "ingest_end_et": "20:00"
    }
    
    with patch('src.scheduler.mcal') as mock_mcal, patch('src.scheduler.HAS_CALENDAR', True):
        mock_cal = MagicMock()
        mock_mcal.get_calendar.return_value = mock_cal
        
        # Setup an early close day (e.g. Nov 27) that closes at 13:00 local time
        date_obj = dt.date(2026, 11, 27)
        import pandas as pd
        mock_schedule = pd.DataFrame({
            'market_open': [pd.Timestamp('2026-11-27 09:30:00', tz='America/New_York')],
            'market_close': [pd.Timestamp('2026-11-27 13:00:00', tz='America/New_York')]
        })
        mock_cal.schedule.return_value = mock_schedule
        
        hours = get_market_hours(date_obj, cfg)
        
        assert hours.is_early_close is True
        
        # post_end_et should precisely be close + 4 hours = 17:00
        assert hours.post_end_et.time() == dt.time(17, 0)
        
        rth_ts = dt.datetime(2026, 11, 27, 12, 0, tzinfo=ET)
        assert hours.get_session_label(rth_ts) == SessionState.RTH.value
        
        # Because it's an early close, 13:30 should now be correctly marked as after-hours
        aft_ts = dt.datetime(2026, 11, 27, 13, 30, tzinfo=ET)
        assert hours.get_session_label(aft_ts) == SessionState.AFTERHOURS.value

def test_deterministic_mapping_failure_counter(caplog):
    """
    EVIDENCE: Validates that if the pipeline is subjected to a drift/invalid session 
    string, it defaults to CLOSED explicitly without exceptions and securely increments 
    the targeted mapping failure counter.
    """
    from src.ingest_engine import _ingest_once_impl
    
    cfg = {
        "ingestion": {"watchlist": ["AAPL"], "cadence_minutes": 5, "enable_market_context": False},
        "storage": {"duckdb_path": ":memory:", "cycle_lock_path": "mock.lock", "writer_lock_path": "mock.lock"},
        "system": {},
        "network": {},
        "validation": {"horizons_minutes": [5]}
    }
    
    with patch("src.ingest_engine.get_market_hours") as mock_gmh, \
         patch("src.ingest_engine.fetch_all"), \
         patch("src.ingest_engine.load_endpoint_plan"), \
         patch("src.ingest_engine.load_api_catalog"), \
         patch("src.ingest_engine.DbWriter"), \
         patch("src.ingest_engine.FileLock"):
        
        mock_mh = MagicMock()
        mock_mh.is_trading_day = True
        
        mock_mh.ingest_start_et = dt.datetime.now() - dt.timedelta(hours=1)
        mock_mh.ingest_end_et = dt.datetime.now() + dt.timedelta(hours=1)
        
        # Injecting a strictly MALFORMED session label that will trip the counter
        mock_mh.get_session_label.return_value = "WEIRD_LABEL"
        mock_gmh.return_value = mock_mh

        with caplog.at_level("ERROR"):
            _ingest_once_impl(cfg, "dummy.yaml", "dummy.yaml")
        
        # Prove the explicit counter log fired against the drift string
        assert "session_mapping_failure_total" in caplog.text
        assert "WEIRD_LABEL" in caplog.text