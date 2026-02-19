# UW Intraday Stack (Unusual Whales) — Audit-First Intraday Analytics

## What this does
- Polls Unusual Whales endpoints **every 5 minutes** during **04:00–20:00 ET** on NYSE trading days (holiday + early-close aware).
- Persists **raw payloads**, **snapshot lineage**, **derived features/levels**, and **probabilistic predictions** to DuckDB.
- Validates predictions walk-forward with no leakage and computes Brier / log loss.
- Applies **guardrailed adaptation** by updating ONLY `src/config/config.yaml` (versioned copy) and logging changes.

## Setup

### 1) Create venv + install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Export your API key
```bash
export UW_API_KEY="YOUR_KEY"
# optional override
export UW_BASE_URL="https://api.unusualwhales.com"
```

### 3) Place your catalog in repo root
The system requires the allowlist file:
- `api_catalog.generated.yaml`

## Run

### One-shot ingest (useful to test auth + schema)
```bash
python -m src.main ingest-once --catalog api_catalog.generated.yaml
```

### Continuous ingest loop (5-min cadence)
```bash
python -m src.main ingest --catalog api_catalog.generated.yaml
```

### Streamlit dashboard
In a second terminal:
```bash
streamlit run src/dashboard_app.py
```

DB file:
- `./uw_stack.duckdb`

## Validation rules (your spec)
Horizons:
- 15 minutes (Tactical)
- 60 minutes (Trend)

Outcome labeling:
- If abs(pct_move) < **0.10%** => **FLAT**
- If pct_move >= 0.10% => **UP**
- If pct_move <= -0.10% => **DOWN**

Configured in:
- `src/config/config.yaml` under `validation`.

## Notes
- All HTTP calls are validated against the catalog (method + path + query params). Unknown endpoints/params hard-fail.
- The writer lock (`uw_stack.duckdb.lock`) prevents accidental multi-writer corruption.


### Concurrency & Locks
- `uw_cycle.lock` prevents multiple ingestors from running the same cycle concurrently (held only during a cycle).
- `uw_stack.duckdb.lock` is an exclusive DB writer lock (held only during DB writes).
- Dashboard and replay use read-only DB connections (no writer lock).


### Validator Hygiene
- Predictions persist `start_price` (spot at decision time) to avoid any dependency on external/live data during validation.
- Validation uses only stored snapshots/features/predictions in DuckDB (no API calls).
