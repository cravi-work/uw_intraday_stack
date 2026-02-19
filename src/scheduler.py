from __future__ import annotations
import datetime as dt
from dataclasses import dataclass
from typing import Dict, Any, Optional
from zoneinfo import ZoneInfo
import logging

try:
    import pandas_market_calendars as mcal
    HAS_CALENDAR = True
except ImportError:
    HAS_CALENDAR = False

logger = logging.getLogger(__name__)
UTC = ZoneInfo("UTC")
ET = ZoneInfo("America/New_York")

@dataclass(frozen=True)
class MarketHours:
    is_trading_day: bool
    is_early_close: bool
    date_et: dt.date
    pre_start_et: Optional[dt.datetime]
    market_open_et: Optional[dt.datetime]
    market_close_et: Optional[dt.datetime]
    post_end_et: Optional[dt.datetime]
    ingest_start_et: Optional[dt.datetime]
    ingest_end_et: Optional[dt.datetime]
    reason: str = "NORMAL"

    def __post_init__(self):
        if self.is_trading_day:
            if not all([
                self.pre_start_et, self.market_open_et, 
                self.market_close_et, self.post_end_et,
                self.ingest_start_et, self.ingest_end_et
            ]):
                raise ValueError(f"Trading day {self.date_et} has incomplete boundaries. Critical config error.")

    def get_session_label(self, ref_et: dt.datetime) -> str:
        if not self.is_trading_day: return "CLOSED"
        if ref_et < self.pre_start_et: return "CLOSED"
        if ref_et < self.market_open_et: return "PRE"
        if ref_et < self.market_close_et: return "REG"
        if ref_et < self.post_end_et: return "AFT"
        return "CLOSED"

    def seconds_to_close(self, ref_et: dt.datetime) -> Optional[int]:
        label = self.get_session_label(ref_et)
        if label == "CLOSED": return None
        target = self.market_close_et if label in ("PRE", "REG") else self.post_end_et
        return max(0, int((target - ref_et).total_seconds())) if target else None

def get_market_hours(asof_date: dt.date, ingestion_cfg: Dict[str, Any]) -> MarketHours:
    # [Issue G] Explicit Fail-Fast or Degraded Mode
    if not HAS_CALENDAR:
        if not ingestion_cfg.get("allow_degraded_calendar", False):
            raise RuntimeError("CRITICAL: pandas_market_calendars missing. Set 'allow_degraded_calendar' to True to force fallback (NOT RECOMMENDED).")
        logger.warning(f"Running in DEGRADED mode for {asof_date} (No Calendar). Holidays ignored.")
        return _build_fallback_hours(asof_date, ingestion_cfg)

    def _p(s): return dt.datetime.strptime(s, "%H:%M").time()
    t_pre = _p(ingestion_cfg["premarket_start_et"])
    t_reg_start = _p(ingestion_cfg["regular_start_et"])
    t_reg_end = _p(ingestion_cfg["regular_end_et"])
    t_post = _p(ingestion_cfg["afterhours_end_et"])
    t_ins = _p(ingestion_cfg["ingest_start_et"])
    t_ine = _p(ingestion_cfg["ingest_end_et"])

    if not (t_pre < t_reg_start < t_reg_end < t_post):
        raise ValueError("Config Error: Time boundaries must be strictly ordered.")
    if not (t_ins < t_ine):
        raise ValueError("Config Error: ingest_start must be before ingest_end.")
    if t_ins < t_pre:
        raise ValueError("Config Error: ingest_start_et must be >= premarket_start_et (otherwise ingestion runs while session is CLOSED).")
    
    # [Issue F] Warn if ingest window exceeds post-market
    if t_ine > t_post:
        logger.warning(f"Config Warning: ingest_end ({t_ine}) is after afterhours_end ({t_post}). Ingestion will be capped.")

    nyse = mcal.get_calendar('NYSE')
    sch = nyse.schedule(start_date=asof_date, end_date=asof_date)
    if sch.empty: return MarketHours(False, False, asof_date, None, None, None, None, None, None, "HOLIDAY")

    row = sch.iloc[0]
    real_open = row['market_open'].to_pydatetime().astimezone(ET)
    real_close = row['market_close'].to_pydatetime().astimezone(ET)
    
    std_close_dt = dt.datetime.combine(asof_date, t_reg_end, tzinfo=ET)
    is_early = real_close < (std_close_dt - dt.timedelta(minutes=5))
    
    dummy = dt.date(2000, 1, 1)
    post_dur = dt.datetime.combine(dummy, t_post) - dt.datetime.combine(dummy, t_reg_end)
    
    base = dt.datetime.combine(asof_date, dt.time(0,0), tzinfo=ET)
    pre_start = base.replace(hour=t_pre.hour, minute=t_pre.minute)
    post_end = real_close + post_dur
    cfg_ingest_end_dt = base.replace(hour=t_ine.hour, minute=t_ine.minute)
    real_ingest_end = min(cfg_ingest_end_dt, post_end)
    real_ingest_start = base.replace(hour=t_ins.hour, minute=t_ins.minute)

    return MarketHours(True, is_early, asof_date, pre_start, real_open, real_close, post_end, real_ingest_start, real_ingest_end, "EARLY_CLOSE" if is_early else "NORMAL")

def _build_fallback_hours(asof_date: dt.date, ingestion_cfg: Dict[str, Any]) -> MarketHours:
    if asof_date.weekday() >= 5:
        return MarketHours(False, False, asof_date, None, None, None, None, None, None, "WEEKEND")

    def _p(s): return dt.datetime.strptime(s, "%H:%M").time()
    t_pre = _p(ingestion_cfg["premarket_start_et"])
    t_reg_start = _p(ingestion_cfg["regular_start_et"])
    t_reg_end = _p(ingestion_cfg["regular_end_et"])
    t_post = _p(ingestion_cfg["afterhours_end_et"])
    t_ins = _p(ingestion_cfg["ingest_start_et"])
    t_ine = _p(ingestion_cfg["ingest_end_et"])

    base = dt.datetime.combine(asof_date, dt.time(0,0), tzinfo=ET)
    pre_start = base.replace(hour=t_pre.hour, minute=t_pre.minute)
    market_open = base.replace(hour=t_reg_start.hour, minute=t_reg_start.minute)
    market_close = base.replace(hour=t_reg_end.hour, minute=t_reg_end.minute)
    post_end = base.replace(hour=t_post.hour, minute=t_post.minute)
    ingest_start = base.replace(hour=t_ins.hour, minute=t_ins.minute)
    ingest_end = min(base.replace(hour=t_ine.hour, minute=t_ine.minute), post_end)

    return MarketHours(True, False, asof_date, pre_start, market_open, market_close, post_end, ingest_start, ingest_end, "DEGRADED_NO_CALENDAR")

def floor_to_interval(dt_obj: dt.datetime, interval_minutes: int) -> dt.datetime:
    return dt_obj - dt.timedelta(minutes=dt_obj.minute % interval_minutes, seconds=dt_obj.second, microseconds=dt_obj.microsecond)