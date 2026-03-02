import datetime as dt
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.models import SessionState
from src.scheduler import ET, HaltState, MarketContext, coerce_session_state, get_market_hours, resolve_market_context

CFG = {
    "premarket_start_et": "04:00",
    "regular_start_et": "09:30",
    "regular_end_et": "16:00",
    "afterhours_end_et": "20:00",
    "ingest_start_et": "04:00",
    "ingest_end_et": "20:00",
    "venue": "NYSE",
    "product": "OPTIONS",
    "calendar_name": "NYSE",
    "timezone_name": "America/New_York",
}


def _mock_schedule(open_ts: str, close_ts: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "market_open": [pd.Timestamp(open_ts, tz="America/New_York")],
            "market_close": [pd.Timestamp(close_ts, tz="America/New_York")],
        }
    )


def test_market_context_metadata_and_session_labels():
    with patch("src.scheduler.mcal") as mock_mcal, patch("src.scheduler.HAS_CALENDAR", True):
        mock_cal = MagicMock()
        mock_mcal.get_calendar.return_value = mock_cal
        mock_cal.schedule.return_value = _mock_schedule("2026-01-05 09:30:00", "2026-01-05 16:00:00")

        ctx = resolve_market_context(dt.date(2026, 1, 5), CFG)
        hours = get_market_hours(dt.date(2026, 1, 5), CFG)

        assert isinstance(ctx, MarketContext)
        assert isinstance(hours, MarketContext)
        assert ctx.venue == "NYSE"
        assert ctx.product == "OPTIONS"
        assert ctx.calendar_name == "NYSE"
        assert ctx.timezone_name == "America/New_York"
        assert ctx.reason == "NORMAL"
        assert ctx.is_early_close is False
        assert ctx.is_half_day is False
        assert ctx.is_holiday is False
        assert ctx.halt_state == HaltState.UNKNOWN

        assert ctx.get_session_label(dt.datetime(2026, 1, 5, 8, 0, tzinfo=ET)) == SessionState.PREMARKET.value
        assert ctx.get_session_label(dt.datetime(2026, 1, 5, 12, 0, tzinfo=ET)) == SessionState.RTH.value
        assert ctx.get_session_label(dt.datetime(2026, 1, 5, 17, 0, tzinfo=ET)) == SessionState.AFTERHOURS.value


def test_half_day_and_early_close_flags_are_explicit():
    with patch("src.scheduler.mcal") as mock_mcal, patch("src.scheduler.HAS_CALENDAR", True):
        mock_cal = MagicMock()
        mock_mcal.get_calendar.return_value = mock_cal
        mock_cal.schedule.return_value = _mock_schedule("2026-11-27 09:30:00", "2026-11-27 13:00:00")

        ctx = resolve_market_context(dt.date(2026, 11, 27), CFG)

        assert ctx.reason == "HALF_DAY"
        assert ctx.is_early_close is True
        assert ctx.is_half_day is True
        assert ctx.post_end_et.time() == dt.time(17, 0)
        assert ctx.ingest_end_et.time() == dt.time(17, 0)
        assert ctx.get_session_label(dt.datetime(2026, 11, 27, 14, 0, tzinfo=ET)) == SessionState.AFTERHOURS.value


def test_holiday_context_is_closed_and_flagged():
    with patch("src.scheduler.mcal") as mock_mcal, patch("src.scheduler.HAS_CALENDAR", True):
        mock_cal = MagicMock()
        mock_mcal.get_calendar.return_value = mock_cal
        mock_cal.schedule.return_value = pd.DataFrame(columns=["market_open", "market_close"])

        ctx = resolve_market_context(dt.date(2026, 1, 19), CFG)

        assert ctx.is_trading_day is False
        assert ctx.reason == "HOLIDAY"
        assert ctx.is_holiday is True
        assert ctx.is_unscheduled_closure is False
        assert ctx.get_session_label(dt.datetime(2026, 1, 19, 12, 0, tzinfo=ET)) == SessionState.CLOSED.value


def test_unscheduled_closure_short_circuits_calendar_resolution():
    cfg = {
        **CFG,
        "unscheduled_closures": [{"date": "2026-01-09", "reason": "UNSCHEDULED_CLOSURE"}],
    }
    with patch("src.scheduler.mcal") as mock_mcal, patch("src.scheduler.HAS_CALENDAR", True):
        ctx = resolve_market_context(dt.date(2026, 1, 9), cfg)

        mock_mcal.get_calendar.assert_not_called()
        assert ctx.is_trading_day is False
        assert ctx.reason == "UNSCHEDULED_CLOSURE"
        assert ctx.is_unscheduled_closure is True
        assert ctx.is_holiday is False


def test_halt_state_override_and_session_coercion():
    cfg = {
        **CFG,
        "halt_states": [{"date": "2026-01-12", "state": "HALTED"}],
    }
    with patch("src.scheduler.mcal") as mock_mcal, patch("src.scheduler.HAS_CALENDAR", True):
        mock_cal = MagicMock()
        mock_mcal.get_calendar.return_value = mock_cal
        mock_cal.schedule.return_value = _mock_schedule("2026-01-12 09:30:00", "2026-01-12 16:00:00")

        ctx = resolve_market_context(dt.date(2026, 1, 12), cfg)

        assert ctx.halt_state == HaltState.HALTED
        assert coerce_session_state("RTH") == SessionState.RTH
        with pytest.raises(ValueError, match="Unknown session label: REG"):
            coerce_session_state("REG")
