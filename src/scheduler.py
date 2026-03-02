import datetime as dt
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Iterable, Optional, Union
from zoneinfo import ZoneInfo

from src.models import SessionState

try:
    import pandas_market_calendars as mcal
    HAS_CALENDAR = True
except ImportError:
    HAS_CALENDAR = False

logger = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")
UTC = dt.timezone.utc


class HaltState(str, Enum):
    NONE = "NONE"
    HALTED = "HALTED"
    UNKNOWN = "UNKNOWN"


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

    def _localize(self, ts: dt.datetime) -> dt.datetime:
        tz_name = getattr(self, "timezone_name", ET.key)
        tz = ZoneInfo(tz_name)
        if ts.tzinfo is None:
            return ts.replace(tzinfo=tz)
        return ts.astimezone(tz)

    def session_state_at(self, now_et: dt.datetime) -> SessionState:
        if not self.is_trading_day:
            return SessionState.CLOSED

        local_now = self._localize(now_et)
        is_pre = (
            self.premarket_start_et
            and self.market_open_et
            and self.premarket_start_et <= local_now < self.market_open_et
        )
        is_reg = (
            self.market_open_et
            and self.market_close_et
            and self.market_open_et <= local_now < self.market_close_et
        )
        is_aft = (
            self.market_close_et
            and self.post_end_et
            and self.market_close_et <= local_now < self.post_end_et
        )

        if is_pre:
            return SessionState.PREMARKET
        if is_reg:
            return SessionState.RTH
        if is_aft:
            return SessionState.AFTERHOURS
        return SessionState.CLOSED

    def seconds_to_close(self, now_et: dt.datetime) -> Optional[int]:
        if not self.is_trading_day or not self.market_close_et or not self.post_end_et:
            return None

        local_now = self._localize(now_et)
        if local_now < self.market_close_et:
            return max(0, int((self.market_close_et - local_now).total_seconds()))
        if local_now < self.post_end_et:
            return max(0, int((self.post_end_et - local_now).total_seconds()))
        return 0

    def get_session_label(self, now_et: dt.datetime) -> str:
        return self.session_state_at(now_et).value


@dataclass
class MarketContext(MarketHours):
    venue: str = "NYSE"
    product: str = "EQUITY"
    calendar_name: str = "NYSE"
    timezone_name: str = ET.key
    is_half_day: bool = False
    is_holiday: bool = False
    is_unscheduled_closure: bool = False
    halt_state: HaltState = HaltState.UNKNOWN
    resolver_mode: str = "CALENDAR"


def floor_to_interval(ts: dt.datetime, minutes: int) -> dt.datetime:
    floored_minute = (ts.minute // minutes) * minutes
    return ts.replace(minute=floored_minute, second=0, microsecond=0)


def coerce_session_state(session_label: str) -> SessionState:
    try:
        return SessionState(session_label)
    except ValueError as exc:
        allowed = ", ".join(s.value for s in SessionState)
        raise ValueError(f"Unknown session label: {session_label}. Allowed: {allowed}") from exc


def _cfg_scope(cfg: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(cfg, dict) and "ingestion" in cfg and isinstance(cfg.get("ingestion"), dict):
        return cfg["ingestion"]
    return cfg


def _resolve_timezone(cfg: Dict[str, Any]) -> ZoneInfo:
    scoped = _cfg_scope(cfg)
    tz_name = (
        scoped.get("timezone_name")
        or scoped.get("timezone")
        or scoped.get("market_timezone")
        or ET.key
    )
    return ZoneInfo(tz_name)


def _resolve_calendar_name(cfg: Dict[str, Any]) -> str:
    scoped = _cfg_scope(cfg)
    return scoped.get("calendar_name") or scoped.get("venue") or "NYSE"


def _resolve_venue(cfg: Dict[str, Any]) -> str:
    return _cfg_scope(cfg).get("venue") or _resolve_calendar_name(cfg)


def _resolve_product(cfg: Dict[str, Any]) -> str:
    return _cfg_scope(cfg).get("product") or "EQUITY"


def _coerce_date(date_obj: Union[dt.date, dt.datetime], tz: ZoneInfo) -> dt.date:
    if isinstance(date_obj, dt.datetime):
        if date_obj.tzinfo is None:
            return date_obj.date()
        return date_obj.astimezone(tz).date()
    return date_obj


def _parse_time(time_str: str, date_obj: dt.date, tz: ZoneInfo = ET) -> dt.datetime:
    t = dt.datetime.strptime(time_str, "%H:%M").time()
    return dt.datetime.combine(date_obj, t, tzinfo=tz)


def _coerce_halt_state(value: Any) -> HaltState:
    if value is None:
        return HaltState.UNKNOWN
    if isinstance(value, HaltState):
        return value

    norm = str(value).strip().upper()
    aliases = {
        "OPEN": HaltState.NONE,
        "CLEAR": HaltState.NONE,
        "TRADING": HaltState.NONE,
        "ACTIVE": HaltState.NONE,
        "NONE": HaltState.NONE,
        "HALTED": HaltState.HALTED,
        "PAUSED": HaltState.HALTED,
        "LULD": HaltState.HALTED,
        "UNKNOWN": HaltState.UNKNOWN,
    }
    return aliases.get(norm, HaltState.UNKNOWN)


def _iter_date_overrides(cfg: Dict[str, Any], key: str) -> Iterable[Any]:
    scoped = _cfg_scope(cfg)
    return scoped.get(key, []) or []


def _lookup_date_override(cfg: Dict[str, Any], key: str, date_obj: dt.date) -> Optional[Dict[str, Any]]:
    target = date_obj.isoformat()
    for entry in _iter_date_overrides(cfg, key):
        if isinstance(entry, str):
            if entry == target:
                return {"date": entry}
            continue
        if isinstance(entry, dict) and entry.get("date") == target:
            return entry
    return None


def _resolve_halt_state_for_date(cfg: Dict[str, Any], date_obj: dt.date) -> HaltState:
    scoped = _cfg_scope(cfg)
    by_date = _lookup_date_override(scoped, "halt_states", date_obj)
    if by_date is not None:
        return _coerce_halt_state(by_date.get("state") or by_date.get("halt_state"))
    return _coerce_halt_state(scoped.get("halt_state"))


def _build_closed_context(
    *,
    cfg: Dict[str, Any],
    reason: str,
    resolver_mode: str,
    is_holiday: bool,
    is_unscheduled_closure: bool,
    halt_state: HaltState,
) -> MarketContext:
    return MarketContext(
        is_trading_day=False,
        reason=reason,
        premarket_start_et=None,
        market_open_et=None,
        market_close_et=None,
        post_end_et=None,
        ingest_start_et=None,
        ingest_end_et=None,
        is_early_close=False,
        venue=_resolve_venue(cfg),
        product=_resolve_product(cfg),
        calendar_name=_resolve_calendar_name(cfg),
        timezone_name=_resolve_timezone(cfg).key,
        is_half_day=False,
        is_holiday=is_holiday,
        is_unscheduled_closure=is_unscheduled_closure,
        halt_state=halt_state,
        resolver_mode=resolver_mode,
    )


def _build_fallback_context(date_obj: dt.date, cfg: Dict[str, Any], reason: str) -> MarketContext:
    scoped = _cfg_scope(cfg)
    if date_obj.weekday() >= 5:
        return _build_closed_context(
            cfg=scoped,
            reason="WEEKEND",
            resolver_mode="FALLBACK",
            is_holiday=False,
            is_unscheduled_closure=False,
            halt_state=_resolve_halt_state_for_date(scoped, date_obj),
        )

    tz = _resolve_timezone(scoped)
    pre_start = _parse_time(scoped["premarket_start_et"], date_obj, tz)
    reg_start = _parse_time(scoped["regular_start_et"], date_obj, tz)
    reg_end = _parse_time(scoped["regular_end_et"], date_obj, tz)
    aft_end = _parse_time(scoped["afterhours_end_et"], date_obj, tz)
    ing_start = _parse_time(scoped["ingest_start_et"], date_obj, tz)
    ing_end = _parse_time(scoped["ingest_end_et"], date_obj, tz)

    if ing_start >= ing_end:
        raise ValueError(f"Config error: ingest_start must be before ingest_end. {ing_start} vs {ing_end}")

    actual_ing_end = min(ing_end, aft_end)

    return MarketContext(
        is_trading_day=True,
        reason=reason,
        premarket_start_et=pre_start,
        market_open_et=reg_start,
        market_close_et=reg_end,
        post_end_et=aft_end,
        ingest_start_et=ing_start,
        ingest_end_et=actual_ing_end,
        is_early_close=False,
        venue=_resolve_venue(scoped),
        product=_resolve_product(scoped),
        calendar_name=_resolve_calendar_name(scoped),
        timezone_name=tz.key,
        is_half_day=False,
        is_holiday=False,
        is_unscheduled_closure=False,
        halt_state=_resolve_halt_state_for_date(scoped, date_obj),
        resolver_mode="FALLBACK",
    )


def resolve_market_context(date_obj: Union[dt.date, dt.datetime], cfg: Dict[str, Any]) -> MarketContext:
    scoped = _cfg_scope(cfg)
    tz = _resolve_timezone(scoped)
    session_date = _coerce_date(date_obj, tz)

    closure_override = _lookup_date_override(scoped, "unscheduled_closures", session_date)
    if closure_override is not None:
        return _build_closed_context(
            cfg=scoped,
            reason=closure_override.get("reason") or "UNSCHEDULED_CLOSURE",
            resolver_mode="OVERRIDE",
            is_holiday=False,
            is_unscheduled_closure=True,
            halt_state=_coerce_halt_state(closure_override.get("halt_state") or closure_override.get("state")),
        )

    if not HAS_CALENDAR:
        if scoped.get("allow_degraded_calendar", False):
            logger.warning(
                "Running in DEGRADED mode for %s (No Calendar). Holidays ignored.",
                session_date,
            )
            return _build_fallback_context(session_date, scoped, "DEGRADED_NO_CALENDAR")
        raise RuntimeError(
            "pandas_market_calendars is required for safe production scheduling. "
            "Set allow_degraded_calendar=True to bypass."
        )

    calendar_name = _resolve_calendar_name(scoped)
    calendar = mcal.get_calendar(calendar_name)
    schedule = calendar.schedule(start_date=session_date, end_date=session_date)

    if schedule.empty:
        reason = "HOLIDAY" if session_date.weekday() < 5 else "WEEKEND"
        return _build_closed_context(
            cfg=scoped,
            reason=reason,
            resolver_mode="CALENDAR",
            is_holiday=(reason == "HOLIDAY"),
            is_unscheduled_closure=False,
            halt_state=_resolve_halt_state_for_date(scoped, session_date),
        )

    market_open = schedule.iloc[0]["market_open"].astimezone(tz)
    market_close = schedule.iloc[0]["market_close"].astimezone(tz)

    pre_start = _parse_time(scoped["premarket_start_et"], session_date, tz)
    regular_end_cfg = _parse_time(scoped["regular_end_et"], session_date, tz)
    afterhours_end_cfg = _parse_time(scoped["afterhours_end_et"], session_date, tz)
    ing_start = _parse_time(scoped["ingest_start_et"], session_date, tz)
    ing_end = _parse_time(scoped["ingest_end_et"], session_date, tz)

    if ing_start >= ing_end:
        raise ValueError(f"Config error: ingest_start must be before ingest_end. {ing_start} vs {ing_end}")

    afterhours_extension = max(dt.timedelta(0), afterhours_end_cfg - regular_end_cfg)
    post_end = market_close + afterhours_extension
    actual_ing_end = min(ing_end, post_end)

    is_early_close = market_close < regular_end_cfg
    session_length = market_close - market_open
    is_half_day = is_early_close and session_length <= dt.timedelta(hours=4)
    reason = "HALF_DAY" if is_half_day else ("EARLY_CLOSE" if is_early_close else "NORMAL")

    return MarketContext(
        is_trading_day=True,
        reason=reason,
        premarket_start_et=pre_start,
        market_open_et=market_open,
        market_close_et=market_close,
        post_end_et=post_end,
        ingest_start_et=ing_start,
        ingest_end_et=actual_ing_end,
        is_early_close=is_early_close,
        venue=_resolve_venue(scoped),
        product=_resolve_product(scoped),
        calendar_name=calendar_name,
        timezone_name=tz.key,
        is_half_day=is_half_day,
        is_holiday=False,
        is_unscheduled_closure=False,
        halt_state=_resolve_halt_state_for_date(scoped, session_date),
        resolver_mode="CALENDAR",
    )


def get_market_hours(date_obj: Union[dt.date, dt.datetime], cfg: Dict[str, Any]) -> MarketHours:
    return resolve_market_context(date_obj, cfg)
