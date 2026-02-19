from __future__ import annotations

import asyncio
import datetime as dt
import logging
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .api_catalog_loader import load_api_catalog
from .config_loader import load_endpoint_plan
from .file_lock import FileLock, FileLockError
from .scheduler import ET, UTC, floor_to_interval, get_market_hours
from .storage import DbWriter
from .uw_client import UwClient

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PlannedCall:
    name: str
    method: str
    path: str
    path_params: Dict[str, Any]
    query_params: Dict[str, Any]
    is_market: bool


def _validate_config(cfg: Dict[str, Any]) -> None:
    req = ["ingestion", "storage", "system", "network", "validation"]
    for s in req:
        if s not in cfg:
            raise KeyError(f"Config missing section: {s}")
    for k in ["duckdb_path", "cycle_lock_path", "writer_lock_path"]:
        if k not in cfg["storage"]:
            raise KeyError(f"Missing storage.{k}")
    if "watchlist" not in cfg["ingestion"]:
        raise KeyError("Missing ingestion.watchlist")
    if "cadence_minutes" not in cfg["ingestion"]:
        raise KeyError("Missing ingestion.cadence_minutes")
    if "horizons_minutes" not in cfg["validation"]:
        raise KeyError("Missing validation.horizons_minutes")


def build_plan(cfg: Dict[str, Any], plan_yaml: Dict[str, Any]) -> Tuple[List[PlannedCall], List[PlannedCall]]:
    def _parse(l, market: bool = False) -> List[PlannedCall]:
        return [
            PlannedCall(
                x["name"],
                x["method"],
                x["path"],
                x.get("path_params", {}) or {},
                x.get("query_params", {}) or {},
                market,
            )
            for x in (l or [])
        ]

    core = _parse(plan_yaml.get("plans", {}).get("default", []))
    market = (
        _parse(plan_yaml.get("plans", {}).get("market_context", []), True)
        if cfg["ingestion"].get("enable_market_context")
        else []
    )
    return core, market


def _expand(call: PlannedCall, ticker: str, date_str: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    def _sub(v):
        if isinstance(v, str):
            return v.replace("{ticker}", ticker).replace("{date}", date_str)
        return v

    path_params = {k: _sub(v) for k, v in call.path_params.items()}
    if "{ticker}" in call.path:
        path_params = {**path_params, "ticker": ticker}

    query_params = {k: _sub(v) for k, v in call.query_params.items() if v is not None}
    return path_params, query_params


def _payload_valid(res) -> bool:
    if not getattr(res, "ok", False):
        return False
    pj = getattr(res, "payload_json", None)
    if pj is None:
        return False
    if isinstance(pj, dict):
        return len(pj) > 0
    if isinstance(pj, list):
        return len(pj) > 0
    return False


async def fetch_all(
    client: UwClient,
    tickers: List[str],
    date_str: str,
    core: List[PlannedCall],
    market: List[PlannedCall],
    *,
    max_concurrency: int,
):
    tasks = []
    sem = asyncio.Semaphore(max(1, int(max_concurrency)))

    async def _one(tkr: str, c: PlannedCall, pp: Dict[str, Any], qp: Dict[str, Any]):
        async with sem:
            sig, res, cb = await client.request(c.method, c.path, path_params=pp, query_params=qp)
            return (tkr, c, sig, qp, res, cb)

    for tkr in sorted(tickers):
        for c in core:
            pp, qp = _expand(c, tkr, date_str)
            tasks.append(_one(tkr, c, pp, qp))

    for c in market:
        pp, qp = _expand(c, "", date_str)
        tasks.append(_one("__MARKET__", c, pp, qp))

    return await asyncio.gather(*tasks)


def _ingest_once_impl(cfg: Dict[str, Any], catalog_path: str) -> None:
    _validate_config(cfg)

    registry = load_api_catalog(catalog_path)
    plan_yaml = load_endpoint_plan("src/config/endpoint_plan.yaml")
    core, market = build_plan(cfg, plan_yaml)
    tickers = [t.upper() for t in cfg["ingestion"]["watchlist"]]

    now_et = dt.datetime.now(ET)
    asof_et = floor_to_interval(now_et, int(cfg["ingestion"]["cadence_minutes"]))
    asof_utc = asof_et.astimezone(UTC)

    hours = get_market_hours(asof_et.date(), cfg["ingestion"])
    if not hours.is_trading_day:
        logger.info("Market Closed", extra={"json": {"reason": hours.reason}})
        return

    if asof_et < hours.ingest_start_et or asof_et >= hours.ingest_end_et:
        logger.info("Outside ingest window", extra={"json": {"asof_et": asof_et.isoformat()}})
        return

    sess = hours.get_session_label(asof_et)
    close_utc = hours.market_close_et.astimezone(UTC) if hours.market_close_et else None
    post_utc = hours.post_end_et.astimezone(UTC) if hours.post_end_et else None
    sec_to_close = hours.seconds_to_close(asof_et)

    # Fetch OUTSIDE the cycle lock (avoid holding lock over network IO).
    async def _run_fetch():
        async with UwClient(registry=registry, **cfg["system"], **cfg["network"]) as client:
            return await fetch_all(
                client,
                tickers,
                asof_et.date().isoformat(),
                core,
                market,
                max_concurrency=cfg.get("network", {}).get("max_concurrency", 20),
            )

    fetch_results = asyncio.run(_run_fetch())

    db = DbWriter(cfg["storage"]["duckdb_path"], cfg["storage"]["writer_lock_path"])

    try:
        # Hold lock ONLY for DB write/commit section.
        with FileLock(cfg["storage"]["cycle_lock_path"]):
            with db.writer() as con:
                db.ensure_schema(con)
                db.upsert_tickers(con, tickers)

                cfg_text = open("src/config/config.yaml", "r", encoding="utf-8").read()
                cfg_ver = db.insert_config(con, cfg_text)

                run_notes = f"SESS={sess}"
                if hours.reason != "NORMAL":
                    run_notes += f"; {hours.reason}"
                run_id = db.begin_run(
                    con,
                    asof_utc,
                    sess,
                    hours.is_trading_day,
                    hours.is_early_close,
                    cfg_ver,
                    registry.catalog_hash,
                    notes=run_notes,
                )

                # Persist events with correct run_id; build per-ticker event lists for DQ/lineage/state.
                events_by_ticker: Dict[str, List[Tuple[int, uuid.UUID, Any]]] = {t: [] for t in tickers}

                for (tkr, call, sig, qp, res, cb) in fetch_results:
                    endpoint_id = db.upsert_endpoint(con, call.method, call.path, qp, registry)
                    event_id = db.insert_raw_event(
                        con,
                        run_id,
                        tkr,
                        endpoint_id,
                        res.requested_at_utc,
                        res.received_at_utc,
                        res.status_code,
                        res.latency_ms,
                        res.payload_hash,
                        res.payload_json,
                        res.retry_count > 0,
                        res.error_type,
                        res.error_message,
                        cb,
                    )

                    if tkr in events_by_ticker:
                        events_by_ticker[tkr].append((endpoint_id, event_id, res))

                    # endpoint_state is per ticker+endpoint; only advance on SUCCESS payloads.
                    if tkr != "__MARKET__" and getattr(res, "ok", False) and res.payload_hash:
                        prev = con.execute(
                            "SELECT last_payload_hash FROM endpoint_state WHERE ticker=? AND endpoint_id=?",
                            [tkr, endpoint_id],
                        ).fetchone()
                        changed = (prev is None) or (prev[0] != res.payload_hash)
                        db.upsert_endpoint_state_success(
                            con, tkr, endpoint_id, event_id, res.received_at_utc, res.payload_hash, changed
                        )

                # Write snapshots + lineage + predictions.
                for tkr in tickers:
                    evs = events_by_ticker.get(tkr, [])
                    valid = sum(1 for (_eid, _event_id, r) in evs if _payload_valid(r))
                    dq = (valid / len(evs)) if evs else 0.0

                    snapshot_id = db.insert_snapshot(
                        con,
                        run_id=run_id,
                        asof_ts_utc=asof_utc,
                        ticker=tkr,
                        session_label=sess,
                        is_trading_day=True,
                        is_early_close=hours.is_early_close,
                        data_quality_score=dq,
                        market_close_utc=close_utc,
                        post_end_utc=post_utc,
                        seconds_to_close=sec_to_close,
                    )

                    # Minimal lineage: associate this snapshot with the latest event for each planned endpoint.
                    for endpoint_id, event_id, r in evs:
                        if _payload_valid(r):
                            freshness = "FRESH"
                        elif getattr(r, "ok", False) and getattr(r, "payload_json", None) is not None:
                            freshness = "INVALID"
                        else:
                            freshness = "MISSING"

                        age = None
                        if getattr(r, "received_at_utc", None):
                            try:
                                age = max(0, int((asof_utc - r.received_at_utc).total_seconds()))
                            except Exception:
                                age = None

                        db.insert_lineage(con, snapshot_id, endpoint_id, event_id, freshness, age)

                    # Mock predictions (until real model is wired in).
                    class MockPred:
                        bias = "NEUTRAL"
                        confidence = 0.0
                        prob_up = 0.0
                        prob_down = 0.0
                        prob_flat = 1.0
                        model_name = "mock"
                        model_version = "1"
                        model_hash = "abc"

                    p = MockPred()
                    start_price = None  # Never invent a price.

                    for h in cfg["validation"]["horizons_minutes"]:
                        db.insert_prediction(
                            con,
                            {
                                "snapshot_id": snapshot_id,
                                "horizon_minutes": int(h),
                                "horizon_kind": "FIXED",
                                "horizon_seconds": None,
                                "start_price": start_price,
                                "bias": p.bias,
                                "confidence": p.confidence,
                                "prob_up": p.prob_up,
                                "prob_down": p.prob_down,
                                "prob_flat": p.prob_flat,
                                "model_name": p.model_name,
                                "model_version": p.model_version,
                                "model_hash": p.model_hash,
                                "is_mock": True,
                            },
                        )

                    if sec_to_close is not None and sec_to_close > 0:
                        db.insert_prediction(
                            con,
                            {
                                "snapshot_id": snapshot_id,
                                "horizon_minutes": 0,  # sentinel for TO_CLOSE
                                "horizon_kind": "TO_CLOSE",
                                "horizon_seconds": int(sec_to_close),
                                "start_price": start_price,
                                "bias": p.bias,
                                "confidence": p.confidence,
                                "prob_up": p.prob_up,
                                "prob_down": p.prob_down,
                                "prob_flat": p.prob_flat,
                                "model_name": p.model_name,
                                "model_version": p.model_version,
                                "model_hash": p.model_hash,
                                "is_mock": True,
                            },
                        )

                db.end_run(con, run_id)

    except FileLockError:
        logger.warning("Skipping cycle: lock held")


class IngestionEngine:
    def __init__(self, *, cfg: Dict[str, Any], catalog_path: str):
        self.cfg = cfg
        self.catalog_path = catalog_path

    def run_cycle(self) -> None:
        _ingest_once_impl(self.cfg, self.catalog_path)
