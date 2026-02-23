from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import duckdb

logger = logging.getLogger(__name__)
UTC = timezone.utc


class CapabilitiesError(Exception):
    """Base class for capabilities-related errors."""
    pass


class DbOpenError(CapabilitiesError):
    """Raised when the database cannot be opened (e.g., locked)."""
    pass


class SchemaMismatchError(CapabilitiesError):
    """Raised when the database schema lacks required tracking columns."""
    pass


def _validate_db_schema(con: duckdb.DuckDBPyConnection) -> None:
    """
    Enforces that the database schema is fully aligned with Phase 0 capabilities requirements.
    Fails fast if older DB schemas are missing required columns.
    """
    required_endpoint_state_columns = [
        "ticker", 
        "endpoint_id", 
        "last_success_event_id", 
        "last_success_ts_utc",
        "last_payload_hash", 
        "last_change_ts_utc", 
        "last_change_event_id",
        "last_attempt_event_id", 
        "last_attempt_ts_utc", 
        "last_attempt_http_status",
        "last_attempt_error_type", 
        "last_attempt_error_msg"
    ]
    
    existing_cols = [r[1] for r in con.execute("PRAGMA table_info('endpoint_state')").fetchall()]
    missing = [c for c in required_endpoint_state_columns if c not in existing_cols]
    
    if missing:
        raise SchemaMismatchError(
            f"Database schema is out of date. Missing columns in 'endpoint_state': {missing}. "
            "Please drop the database and let ingestion recreate it, or run the additive migration."
        )


def _safe_age_seconds(ts: Optional[datetime], now_utc: datetime) -> Optional[int]:
    """Deterministically calculates age in seconds from a UTC timestamp."""
    if not ts:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    return max(0, int((now_utc - ts).total_seconds()))


def extract_db_truth(con: duckdb.DuckDBPyConnection) -> Dict[str, Any]:
    """
    Extracts the exact physical state of the data platform per ticker and endpoint.
    Maintains strict separation between Fetch Age, Success Age, and Payload Change Age.
    """
    now_utc = datetime.now(UTC)
    
    query = """
        SELECT 
            e.ticker,
            d.method,
            d.path,
            e.last_attempt_ts_utc,
            e.last_success_ts_utc,
            e.last_change_ts_utc,
            e.last_attempt_http_status,
            e.last_attempt_error_type,
            e.last_attempt_error_msg
        FROM endpoint_state e
        JOIN dim_endpoints d ON e.endpoint_id = d.endpoint_id
        ORDER BY e.ticker, d.path
    """
    
    rows = con.execute(query).fetchall()
    
    results: Dict[str, Dict[str, Any]] = {}
    
    for row in rows:
        (ticker, method, path, attempt_ts, success_ts, change_ts, 
         http_status, err_type, err_msg) = row
         
        ticker = str(ticker).upper()
        if ticker not in results:
            results[ticker] = {"endpoints": {}}
            
        endpoint_key = f"{method} {path}"
        
        attempt_age = _safe_age_seconds(attempt_ts, now_utc)
        success_age = _safe_age_seconds(success_ts, now_utc)
        change_age = _safe_age_seconds(change_ts, now_utc)
        
        health_status = "ERROR"
        
        if success_age is not None and attempt_age is not None:
            # Endpoint is only considered responsive if its last attempt was successful
            if (http_status is not None and 200 <= http_status < 300) and (err_type is None):
                
                # Dynamic Endpoint-specific SLAs
                # Market price updates should be strictly fresh (e.g. 60s)
                # Options flows can safely drift longer based on regimes (e.g. 90m = 5400s)
                allowed_sla = 60 if "ohlc" in path else 5400
                
                if change_age is not None and change_age > allowed_sla: 
                    health_status = "STALE"
                else:
                    health_status = "HEALTHY"

        results[ticker]["endpoints"][endpoint_key] = {
            "health_status": health_status,
            "attempt_age_s": attempt_age,
            "success_age_s": success_age,
            "payload_change_age_s": change_age,
            "last_http_status": http_status,
            "error_type": err_type,
            "error_message": err_msg
        }
        
    return results


def print_text_report(truth_data: Dict[str, Any]) -> None:
    """Formats the capabilities into a human-readable CLI report without literal evaluation bugs."""
    if not truth_data:
        print("No capability data found in the database. Run an ingestion cycle first.")
        return
        
    for ticker, data in truth_data.items():
        if ticker == "__MARKET__":
            print("\n=== MARKET CONTEXT ENDPOINTS ===")
        else:
            print(f"\n=== TICKER: {ticker} ===")
            
        endpoints = data.get("endpoints", {})
        
        if not endpoints:
            print("  (No endpoint data)")
            continue
            
        for ep_key, stats in endpoints.items():
            hs = stats.get("health_status", "ERROR")
            status_mark = "✅" if hs == "HEALTHY" else "⚠️" if hs == "STALE" else "❌"
            
            att = stats["attempt_age_s"]
            succ = stats["success_age_s"]
            chg = stats["payload_change_age_s"]
            
            # Safe formatting to avoid the literal template evaluation bugs
            att_str = f"{att}s" if att is not None else "N/A"
            succ_str = f"{succ}s" if succ is not None else "N/A"
            chg_str = f"{chg}s" if chg is not None else "N/A"
            
            print(f"{status_mark} [{hs}] {ep_key}")
            print(f"    Attempt Age: {att_str} | Success Age: {succ_str} | Payload Change Age: {chg_str}")
            
            if hs == "ERROR":
                err = stats["error_type"] or "Unknown Error"
                code = stats["last_http_status"] or "N/A"
                msg = stats["error_message"] or ""
                print(f"    └─ Alert: HTTP {code} - {err} ({msg})")


def check_capabilities(catalog_path: str, db_path: str, plan_path: str, output_format: str = "text") -> None:
    """
    Main entry point for verifying system capabilities against the persisted truth model.
    """
    try:
        # Read-only connection to prevent blocking the ingestion writer
        con = duckdb.connect(db_path, read_only=True)
    except duckdb.IOException as e:
        raise DbOpenError(f"Could not open database at {db_path}. Is it locked by another process? Error: {e}")
    
    try:
        _validate_db_schema(con)
        truth_data = extract_db_truth(con)
        
        if output_format == "json":
            print(json.dumps(truth_data, indent=2))
        else:
            print_text_report(truth_data)
            
    finally:
        con.close()