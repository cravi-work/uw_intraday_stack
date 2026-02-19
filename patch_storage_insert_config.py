import pathlib

p = pathlib.Path(r"src\storage.py")
txt = p.read_text(encoding="utf-8")

if "def insert_config" in txt or "DbWriter.insert_config" in txt:
    print("OK: src/storage.py already has insert_config.")
    raise SystemExit(0)

shim = r"""

# --- Compatibility shim: DbWriter.insert_config -------------------------------
# ingest_engine.py expects: db.insert_config(con, config_yaml_text) -> config_version
import hashlib as _hashlib

def _dbwriter_insert_config(self, con, config_yaml_text: str) -> int:
    if config_yaml_text is None or not str(config_yaml_text).strip():
        raise ValueError("config_yaml_text is required")

    txt_norm = str(config_yaml_text).replace("\\r\\n", "\\n").strip() + "\\n"
    h = _hashlib.sha256(txt_norm.encode("utf-8")).hexdigest()

    con.execute(\"\"\"
        CREATE TABLE IF NOT EXISTS meta_config(
            config_version INTEGER,
            config_hash TEXT UNIQUE,
            config_yaml TEXT,
            created_at_utc TIMESTAMP DEFAULT current_timestamp
        )
    \"\"\")

    row = con.execute("SELECT config_version FROM meta_config WHERE config_hash = ?", [h]).fetchone()
    if row:
        return int(row[0])

    mx = con.execute("SELECT COALESCE(MAX(config_version), 0) FROM meta_config").fetchone()[0]
    next_ver = int(mx) + 1

    con.execute(
        "INSERT INTO meta_config(config_version, config_hash, config_yaml) VALUES (?, ?, ?)",
        [next_ver, h, txt_norm],
    )
    return next_ver

try:
    DbWriter.insert_config = _dbwriter_insert_config  # type: ignore[name-defined]
except Exception:
    pass
# --- End shim -----------------------------------------------------------------
"""

p.write_text(txt + shim, encoding="utf-8")
print("PATCHED: appended DbWriter.insert_config shim to src/storage.py")
