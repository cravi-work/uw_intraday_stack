from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
import time
from typing import Any
import duckdb

from .config_loader import load_yaml
from .logging_config import configure_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    subparsers = ap.add_subparsers(dest="cmd", required=True, help="Command to run")

    p_ingest = subparsers.add_parser("ingest", help="Run ingestion loop")
    p_ingest.add_argument("--catalog", default="api_catalog.generated.yaml")
    p_ingest.add_argument("--config", default="src/config/config.yaml")
    p_ingest.add_argument("--log-level", default="INFO")

    p_once = subparsers.add_parser("ingest-once", help="Run single ingestion cycle")
    p_once.add_argument("--catalog", default="api_catalog.generated.yaml")
    p_once.add_argument("--config", default="src/config/config.yaml")
    p_once.add_argument("--log-level", default="INFO")

    p_cap = subparsers.add_parser("capabilities", help="Check endpoint capabilities")
    p_cap.add_argument("--catalog", default="api_catalog.generated.yaml")
    p_cap.add_argument("--db", required=True, help="Path to DuckDB database")
    p_cap.add_argument("--plan", default="src/config/endpoint_plan.yaml")
    p_cap.add_argument("--format", choices=["text", "json"], default="text")
    p_cap.add_argument("--log-level", default="INFO")
    
    p_val = subparsers.add_parser("validate", help="Run validation process over pending predictions")
    p_val.add_argument("--config", default="src/config/config.yaml")
    p_val.add_argument("--log-level", default="INFO")
    
    p_rep = subparsers.add_parser("replay", help="Replay historical snapshots")
    p_rep.add_argument("ticker", type=str, help="Ticker to replay")
    p_rep.add_argument("--log-level", default="INFO")

    return ap.parse_args()


def run_loop(engine: Any) -> None:
    from .scheduler import ET, get_market_hours, floor_to_interval

    while True:
        now_et = dt.datetime.now(ET)
        
        cfg_ingestion = engine.cfg["ingestion"]
        hours = get_market_hours(now_et.date(), cfg_ingestion)

        if hours.is_trading_day and hours.ingest_start_et and hours.ingest_end_et:
            if hours.ingest_start_et <= now_et < hours.ingest_end_et:
                engine.run_cycle()

        cadence_minutes = int(cfg_ingestion["cadence_minutes"])
        floored = floor_to_interval(now_et, cadence_minutes)
        wake = floored + dt.timedelta(minutes=cadence_minutes)
        
        sleep_seconds = max(0.5, (wake - now_et).total_seconds())
        logger.info(
            f"Sleeping until {wake.time()} ET ({int(sleep_seconds)}s)", 
            extra={"json": {"wake_et": wake.isoformat()}}
        )
        
        time.sleep(sleep_seconds)


def main() -> None:
    args = parse_args()
    configure_logging(service="uw_intraday_stack", level=args.log_level)

    if args.cmd == "capabilities":
        from .capabilities import check_capabilities, CapabilitiesError, DbOpenError, SchemaMismatchError
        try:
            check_capabilities(args.catalog, args.db, args.plan, args.format)
            sys.exit(0)
        except SchemaMismatchError as ce:
            print(f"Capabilities Schema Error: {ce}", file=sys.stderr)
            sys.exit(2)
        except DbOpenError as ce:
            print(f"Capabilities Database Error: {ce}", file=sys.stderr)
            sys.exit(2)
        except CapabilitiesError as ce:
            print(f"Capabilities Error: {ce}", file=sys.stderr)
            sys.exit(2)
        except Exception as e:
            print(f"Unexpected Runtime Error: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.cmd == "validate":
        from .validator import validate_pending
        cfg = load_yaml(args.config).raw
        db_path = cfg["storage"]["duckdb_path"]
        con = duckdb.connect(db_path)
        try:
            res = validate_pending(
                con, 
                now_utc=dt.datetime.now(dt.timezone.utc), 
                flat_threshold_pct=cfg.get("validation", {}).get("flat_threshold_pct", 0.001),
                tolerance_minutes=cfg.get("validation", {}).get("tolerance_minutes", 10)
            )
            print(f"Validation successful: {res.updated} updated, {res.skipped} skipped.")
            sys.exit(0)
        except Exception as e:
            print(f"Validation Error: {e}", file=sys.stderr)
            sys.exit(1)
            
    elif args.cmd == "replay":
        from .replay_engine import run_replay
        run_replay(args.ticker)
        sys.exit(0)

    from .ingest_engine import IngestionEngine
    cfg = load_yaml(getattr(args, "config", "src/config/config.yaml")).raw
    engine = IngestionEngine(cfg=cfg, catalog_path=args.catalog, config_path=getattr(args, "config", "src/config/config.yaml"))

    if args.cmd == "ingest-once":
        engine.run_cycle()
    elif args.cmd == "ingest":
        run_loop(engine)


if __name__ == "__main__":
    main()