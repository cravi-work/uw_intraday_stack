# tests/test_cl02_session_mapping.py
import pytest
import datetime as dt
from unittest.mock import MagicMock, patch
from src.models import SessionState
from src.scheduler import get_market_hours, ET

def test_premarket_and_rth_mapping():
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
    from src.ingest_engine import IngestionEngine

    cfg = {
        "ingestion": {"watchlist": ["AAPL"], "cadence_minutes": 5, "enable_market_context": False},
        "storage": {"duckdb_path": ":memory:", "cycle_lock_path": "mock.lock", "writer_lock_path": "mock.lock"},
        "system": {},
        "network": {},
        "validation": {
            "invalid_after_minutes": 60,
            "fallback_max_age_minutes": 15,
            "alignment_tolerance_sec": 900,
            "use_default_required_features": False,
            "emit_to_close_horizon": False,
            "horizon_weights_source": "explicit",
            "horizons_minutes": [5],
            "horizon_critical_features": {"5": []},
            "horizon_weights": {"5": {"spot": 1.0}}
        }
    }

    with patch("src.ingest_engine.get_market_hours") as mock_gmh, \
         patch("src.ingest_engine.fetch_all") as mock_fetch, \
         patch("src.ingest_engine.load_endpoint_plan") as mock_lep, \
         patch("src.ingest_engine.validate_plan_coverage") as mock_vpc, \
         patch("src.ingest_engine.load_api_catalog") as mock_lac:
        
        mock_mh = MagicMock()
        mock_mh.is_trading_day = False
        mock_mh.reason = "HOLIDAY"
        mock_mh.get_session_label.return_value = SessionState.CLOSED.value
        mock_gmh.return_value = mock_mh

        engine = IngestionEngine(cfg=cfg, catalog_path="api_catalog.generated.yaml", config_path="dummy.yaml")
        engine.run_cycle()

        mock_fetch.assert_not_called()

def test_early_close_computes_correct_post_market():
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
        
        date_obj = dt.date(2026, 11, 27)
        import pandas as pd
        mock_schedule = pd.DataFrame({
            'market_open': [pd.Timestamp('2026-11-27 09:30:00', tz='America/New_York')],
            'market_close': [pd.Timestamp('2026-11-27 13:00:00', tz='America/New_York')]
        })
        mock_cal.schedule.return_value = mock_schedule
        
        hours = get_market_hours(date_obj, cfg)
        
        assert hours.is_early_close is True
        assert hours.post_end_et.time() == dt.time(17, 0)
        
        rth_ts = dt.datetime(2026, 11, 27, 12, 0, tzinfo=ET)
        assert hours.get_session_label(rth_ts) == SessionState.RTH.value
        
        aft_ts = dt.datetime(2026, 11, 27, 13, 30, tzinfo=ET)
        assert hours.get_session_label(aft_ts) == SessionState.AFTERHOURS.value

def test_deterministic_mapping_failure_counter(caplog):
    from src.ingest_engine import _ingest_once_impl
    
    cfg = {
        "ingestion": {"watchlist": ["AAPL"], "cadence_minutes": 5, "enable_market_context": False},
        "storage": {"duckdb_path": ":memory:", "cycle_lock_path": "mock.lock", "writer_lock_path": "mock.lock"},
        "system": {},
        "network": {},
        "validation": {
            "invalid_after_minutes": 60,
            "fallback_max_age_minutes": 15,
            "alignment_tolerance_sec": 900,
            "use_default_required_features": False,
            "emit_to_close_horizon": False,
            "horizon_weights_source": "explicit",
            "horizons_minutes": [5],
            "horizon_critical_features": {"5": []},
            "horizon_weights": {"5": {"spot": 1.0}}
        }
    }
    
    with patch("src.ingest_engine.get_market_hours") as mock_gmh, \
         patch("src.ingest_engine.fetch_all") as mock_fetch, \
         patch("src.ingest_engine.load_endpoint_plan") as mock_lep, \
         patch("src.ingest_engine.validate_plan_coverage") as mock_vpc, \
         patch("src.ingest_engine.DbWriter") as mock_dbw_cls, \
         patch("src.ingest_engine.FileLock") as mock_fl, \
         patch("src.ingest_engine.extract_all") as mock_extract:
        
        mock_mh = MagicMock()
        mock_mh.is_trading_day = True
        
        mock_mh.ingest_start_et = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=1)
        mock_mh.ingest_end_et = dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=1)
        mock_mh.get_session_label.return_value = "WEIRD_LABEL"
        mock_gmh.return_value = mock_mh

        mock_lep.return_value = {"plans": {"default": []}}
        mock_extract.return_value = ([], [])

        mock_res = MagicMock()
        mock_res.requested_at_utc = dt.datetime.now(dt.timezone.utc).timestamp()
        mock_res.received_at_utc = mock_res.requested_at_utc + 0.1
        mock_res.status_code = 200
        mock_res.payload_hash = "mock_hash"
        mock_res.payload_json = []
        mock_res.retry_count = 0
        mock_res.error_type = None
        mock_res.error_message = None

        mock_call = MagicMock()
        mock_call.method = "GET"
        mock_call.path = "/api/mock"

        async def fake_fetch(*args, **kwargs):
            return [("AAPL", mock_call, "sig", {}, mock_res, None)]
        mock_fetch.side_effect = fake_fetch

        with caplog.at_level("ERROR"):
            _ingest_once_impl(cfg, "api_catalog.generated.yaml", "dummy.yaml")
        
        assert "Session contract violation" in caplog.text
        assert "WEIRD_LABEL" in caplog.text