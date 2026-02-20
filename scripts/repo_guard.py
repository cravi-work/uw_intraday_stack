#!/usr/bin/env python3
import fnmatch
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, List

@dataclass(frozen=True)
class Rule:
    name: str
    kind: Literal["glob", "exact"]
    pattern: str
    rationale: str

RULES = [
    Rule("DuckDB Files", "glob", "*.duckdb*", "Database artifacts must not be tracked"),
    Rule("DB Backups", "glob", "*.bak", "Database backups must not be tracked"),
    Rule("Lock Files", "glob", "*.lock", "Runtime locks must not be tracked"),
    Rule("Env Files", "glob", ".env*", "Secrets must not be tracked"),
    Rule("Config JSON", "exact", "config.json", "Config secrets must not be tracked"),
    Rule("Cache", "glob", "__pycache__/*", "Cache must not be tracked"),
    Rule("Pyc", "glob", "*.pyc", "Compiled python files must not be tracked"),
    Rule("Logs", "glob", "*.log", "Logs must not be tracked"),
    Rule("Derived Output", "glob", "derived_levels/*", "Generated outputs must not be tracked"),
    Rule("Junk List", "exact", "list[str]", "Accidental type-name file"),
    Rule("Junk Optional", "exact", "optional[any]", "Accidental type-name file"),
    Rule("Junk DuckDBConn", "exact", "duckdb.duckdbpyconnection", "Accidental type-name file"),
    Rule("Junk UUID", "exact", "uuid.uuid", "Accidental type-name file"),
    Rule("Junk int", "exact", "int", "Accidental type-name file"),
    Rule("Junk None", "exact", "none", "Accidental type-name file"),
]

def get_tracked_files() -> List[str]:
    try:
        result = subprocess.run(["git", "ls-files", "-z"], check=True, capture_output=True, text=False)
        return [f for f in result.stdout.decode('utf-8', errors='replace').split('\0') if f]
    except subprocess.CalledProcessError as e:
        print(f"Error executing git ls-files: {e}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print("Error: git command not found.", file=sys.stderr)
        sys.exit(1)

def run_guard() -> None:
    tracked_files = get_tracked_files()
    violations = []
    for raw_path in tracked_files:
        normalized_path = Path(raw_path).as_posix().lower()
        for rule in RULES:
            pattern = rule.pattern.lower()
            if rule.kind == "exact" and (normalized_path == pattern or normalized_path.endswith("/" + pattern)):
                violations.append((raw_path, rule)); break
            elif rule.kind == "glob" and (fnmatch.fnmatch(normalized_path, pattern) or fnmatch.fnmatch(normalized_path, "*/" + pattern)):
                violations.append((raw_path, rule)); break

    if violations:
        print("=" * 80 + "\n[FAIL] REPO GUARD FAILED: Forbidden tracked artifacts found!")
        for file_path, rule in violations: print(f"File: {file_path}\n  Rule: {rule.name} ('{rule.pattern}')\n")
        sys.exit(2)
    print("[PASS] Repo Guard passed. No forbidden artifacts found in index.")
    sys.exit(0)

if __name__ == "__main__":
    run_guard()