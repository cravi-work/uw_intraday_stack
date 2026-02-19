import pathlib, hashlib

PATH = pathlib.Path(r"src\storage.py")

needle = "def insert_config("
marker = "    # ---------------------------\n    # Dimensions / registry helpers"

txt = PATH.read_text(encoding="utf-8")

if needle in txt:
    print("OK: insert_config already exists in src/storage.py")
    raise SystemExit(0)

idx = txt.find(marker)
if idx < 0:
    raise SystemExit("ERROR: Could not find insertion marker in src/storage.py")

method = r'''
    def insert_config(self, con, config_yaml_text: str) -> int:
        """
        Insert the current config YAML into meta_config and return config_version.
        Deterministic hashing: normalize newlines, strip trailing whitespace, ensure trailing newline.
        """
        if config_yaml_text is None or not str(config_yaml_text).strip():
            raise ValueError("config_yaml_text is required")

        txt_norm = str(config_yaml_text).replace("\r\n", "\n").strip() + "\n"
        h = hashlib.sha256(txt_norm.encode("utf-8")).hexdigest()

        row = con.execute(
            "SELECT config_version FROM meta_config WHERE config_hash = ?",
            [h],
        ).fetchone()
        if row:
            return int(row[0])

        con.execute(
            "INSERT INTO meta_config(config_hash, config_yaml) VALUES (?, ?)",
            [h, txt_norm],
        )

        ver = con.execute(
            "SELECT config_version FROM meta_config WHERE config_hash = ?",
            [h],
        ).fetchone()[0]
        return int(ver)

'''
txt2 = txt[:idx] + method + "\n" + txt[idx:]
PATH.write_text(txt2, encoding="utf-8")
print("PATCHED: Added DbWriter.insert_config() to src/storage.py")
^Z

.\.venv\Scripts\python.exe patch_add_insert_config.py
