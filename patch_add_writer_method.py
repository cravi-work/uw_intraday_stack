import pathlib, re

p = pathlib.Path(r"src\storage.py")
txt = p.read_text(encoding="utf-8")

if re.search(r"def\s+writer\s*\(", txt):
    print("OK: DbWriter.writer() already exists")
    raise SystemExit(0)

# Insert writer() just before the schema/migrations section ends.
# We'll place it right after the dataclass definition header + before ensure_schema to be safe.
pattern = r"@dataclass\(frozen=True\)\s*\nclass DbWriter:[\s\S]*?\n\s*def ensure_schema\("
m = re.search(pattern, txt)
if not m:
    raise SystemExit("ERROR: Could not find insertion point near DbWriter.ensure_schema")

insert_at = m.start()

# Build a minimal DbWriter with writer() contextmanager mixin by inserting method right before ensure_schema.
# We inject right before the 'def ensure_schema(' line.
# Find the exact 'def ensure_schema(' position.
pos = txt.find("def ensure_schema", insert_at)
if pos < 0:
    raise SystemExit("ERROR: ensure_schema not found")

method = r'''
    def writer(self):
        """
        Context manager yielding a DuckDB connection for a single write unit.
        Locking is handled externally (main loop). This is just connection lifecycle.
        """
        class _WriterCtx:
            def __init__(self, path: str):
                self._path = path
                self._con = None
            def __enter__(self):
                self._con = duckdb.connect(self._path)
                return self._con
            def __exit__(self, exc_type, exc, tb):
                try:
                    if self._con is not None:
                        self._con.close()
                finally:
                    self._con = None
                return False
        return _WriterCtx(self.duckdb_path)

'''
txt2 = txt[:pos] + method + txt[pos:]
p.write_text(txt2, encoding="utf-8")
print("PATCHED: added DbWriter.writer() to src/storage.py")