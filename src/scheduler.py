import datetime as dt
from zoneinfo import ZoneInfo
from dataclasses import dataclass
from typing import Dict, Any, Optional

from src.models import SessionState

try:
    import pandas_market_calendars as mcal
    HAS_CALENDAR = True
except ImportError:
    HAS_CALENDAR = False

ET = ZoneInfo("America/New_York")
UTC = dt.timezone.utc

@dataclass
class MarketHours:
    is_trading_day: bool
    reason: str
    premarket_start_et: Optional[dt.datetime]
    market_open_et: Optional[dt.datetime]
    market_close_et: Optional[dt.datetime]
    post_end_et: Optional[dt.datetime]
    ingest_start_et: Optional[dt.datetime]
    ingest_end_et: Optional[dt.datetime]
    is_early_close: bool

    def seconds_to_close(self, now_et: dt.datetime) -> Optional[int]:
        if not self.is_trading_day or not self.market_close_et or not self.post_end_et:
            return None
            
        if now_et < self.market_close_et:
            return max(0, int((self.market_close_et - now_et).total_seconds()))
        if now_et < self.post_end_et:
            return max(0, int((self.post_end_et - now_et).total_seconds()))
        return 0
        
    def get_session_label(self, now_et: dt.datetime) -> str:
        if not self.is_trading_day:
            return SessionState.CLOSED.value
            
        is_pre = self.premarket_start_et and self.market_open_et and self.premarket_start_et <= now_et < self.market_open_et
        is_reg = self.market_open_et and self.market_close_et and self.market_open_et <= now_et < self.market_close_et
        is_aft = self.market_close_et and self.post_end_et and self.market_close_et <= now_et < self.post_end_et
        
        if is_pre: return SessionState.PREMARKET.value
        if is_reg: return SessionState.RTH.value
        if is_aft: return SessionState.AFTERHOURS.value
        
        return SessionState.CLOSED.value


def floor_to_interval(ts: dt.datetime, minutes: int) -> dt.datetime:
    floored_minute = (ts.minute // minutes) * minutes
    return ts.replace(minute=floored_minute, second=0, microsecond=0)


def _parse_time(time_str: str, date_obj: dt.date) -> dt.datetime:
    t = dt.datetime.strptime(time_str, "%H:%M").time()
    return dt.datetime.combine(date_obj, t, tzinfo=ET)


def _build_fallback_hours(date_obj: dt.date, cfg: Dict[str, Any], reason: str) -> MarketHours:
    if date_obj.weekday() >= 5:
        return MarketHours(False, "WEEKEND", None, None, None, None, None, None, False)
        
    pre_start = _parse_time(cfg["premarket_start_et"], date_obj)
    reg_start = _parse_time(cfg["regular_start_et"], date_obj)
    reg_end = _parse_time(cfg["regular_end_et"], date_obj)
    aft_end = _parse_time(cfg["afterhours_end_et"], date_obj)
    ing_start = _parse_time(cfg["ingest_start_et"], date_obj)
    ing_end = _parse_time(cfg["ingest_end_et"], date_obj)
    
    if ing_start >= ing_end:
        raise ValueError(f"Config error: ingest_start must be before ingest_end. {ing_start} vs {ing_end}")
        
    actual_ing_end = min(ing_end, aft_end)

    return MarketHours(
        is_trading_day=True,
        reason=reason,
        premarket_start_et=pre_start,
        market_open_et=reg_start,
        market_close_et=reg_end,
        post_end_et=aft_end,
        ingest_start_et=ing_start,
        ingest_end_et=actual_ing_end,
        is_early_close=False
    )

def get_market_hours(date_obj: dt.date, cfg: Dict[str, Any]) -> MarketHours:
    if not HAS_CALENDAR:
        if cfg.get("allow_degraded_calendar", False):
            import logging
            logging.getLogger(__name__).warning(f"Running in DEGRADED mode for {date_obj} (No Calendar). Holidays ignored.")
            return _build_fallback_hours(date_obj, cfg, "DEGRADED_NO_CALENDAR")
        raise RuntimeError("pandas_market_calendars is required for safe production scheduling. Set allow_degraded_calendar=True to bypass.")

    nyse = mcal.get_calendar('NYSE')
    schedule = nyse.schedule(start_date=date_obj, end_date=date_obj)

    if schedule.empty:
        reason = "HOLIDAY" if date_obj.weekday() < 5 else "WEEKEND"
        return MarketHours(False, reason, None, None, None, None, None, None, False)

    market_open = schedule.iloc[0]['market_open'].astimezone(ET)
    market_close = schedule.iloc[0]['market_close'].astimezone(ET)

    pre_start = _parse_time(cfg["premarket_start_et"], date_obj)
    aft_end = _parse_time(cfg["afterhours_end_et"], date_obj)
    ing_start = _parse_time(cfg["ingest_start_et"], date_obj)
    ing_end = _parse_time(cfg["ingest_end_et"], date_obj)

    is_early_close = market_close.time() < dt.time(16, 0)
    
    if ing_start >= ing_end:
        raise ValueError(f"Config error: ingest_start must be before ingest_end. {ing_start} vs {ing_end}")

    actual_ing_end = min(ing_end, aft_end) if not is_early_close else min(ing_end, market_close)

    return MarketHours(
        is_trading_day=True,
        reason="NORMAL",
        premarket_start_et=pre_start,
        market_open_et=market_open,
        market_close_et=market_close,
        post_end_et=aft_end,
        ingest_start_et=ing_start,
        ingest_end_et=actual_ing_end,
        is_early_close=is_early_close
    )