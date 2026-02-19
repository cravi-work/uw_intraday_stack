from __future__ import annotations

import argparse
import datetime as dt
import logging
import time
from typing import Any

from .config_loader import load_yaml
from .logging_config import configure_logging
from .scheduler import ET, in_ingest_window, nyse_market_hours, next_wakeup_et

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    subparsers = ap.add_subparsers(dest="cmd", required=True, help="Command to run")

    # Subparser for 'ingest'
    p_ingest = subparsers.add_parser("ingest", help="Run ingestion loop")
    p_ingest.add_argument("--catalog", default="api_catalog.generated.yaml")
    p_ingest.add_argument("--log-level", default="INFO")

    # Subparser for 'ingest-once'
    p_once = subparsers.add_parser("ingest-once", help="Run single ingestion cycle")
    p_once.add_argument("--catalog", default="api_catalog.generated.yaml")
    p_once.add_argument("--log-level", default="INFO")

    # Subparser for 'capabilities'
    p_cap = subparsers.add_parser("capabilities", help="Check endpoint capabilities")
    p_cap.add_argument("--catalog", default="api_catalog.generated.yaml")
    p_cap.add_argument("--db", required=True, help="Path to DuckDB database")
    p_cap.add_argument("--plan", default="src/config/endpoint_plan.yaml")
    p_cap.add_argument("--format", choices=["text", "json"], default="text")
    p_cap.add_argument("--log-level", default="INFO")

    return ap.parse_args()


def run_loop(engine: Any) -> None:
    # engine is typed as Any to avoid top-level import of IngestionEngine
    while True:
        now_et = dt.datetime.now(ET)
        hours = nyse_market_hours(
            now_et,
            ingest_start_et=engine.cfg["ingestion"]["ingest_start_et"],
            ingest_end_et=engine.cfg["ingestion"]["ingest_end_et"],
            premarket_start_et=engine.cfg["ingestion"]["premarket_start_et"],
            regular_start_et=engine.cfg["ingestion"]["regular_start_et"],
            regular_end_et=engine.cfg["ingestion"]["regular_end_et"],
            afterhours_end_et=engine.cfg["ingestion"]["afterhours_end_et"],
        )

        if in_ingest_window(now_et, hours):
            engine.run_cycle()

        # Cadence may be updated by adapter (via config hot-reload). Use latest value each loop.
        cadence_minutes = int(engine.cfg["ingestion"]["cadence_minutes"])
        wake = next_wakeup_et(now_et, cadence_minutes)
        
        sleep_seconds = max(0.5, (wake - now_et).total_seconds())
        logger.info(f"Sleeping until {wake.time()} ET ({int(sleep_seconds)}s)", extra={"json": {"wake_et": wake.isoformat()}})
        
        time.sleep(sleep_seconds)


def main() -> None:
    args = parse_args()
    configure_logging(service="uw_intraday_stack", level=args.log_level)

    if args.cmd == "capabilities":
        from .capabilities import check_capabilities
        check_capabilities(args.catalog, args.db, args.plan, args.format)
        return

    # Lazy import to keep capabilities command lightweight
    from .ingest_engine import IngestionEngine
    
    cfg = load_yaml("src/config/config.yaml").raw
    engine = IngestionEngine(cfg=cfg, catalog_path=args.catalog)

    if args.cmd == "ingest-once":
        engine.run_cycle()
    elif args.cmd == "ingest":
        run_loop(engine)


if __name__ == "__main__":
    main()
