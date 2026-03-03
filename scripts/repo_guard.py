#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fnmatch
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "venv",
    "env",
    "__pycache__",
}

TEXT_SUFFIXES = {
    "",
    ".bat",
    ".cfg",
    ".csv",
    ".gitignore",
    ".ini",
    ".json",
    ".md",
    ".ps1",
    ".py",
    ".sh",
    ".sql",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}


@dataclass(frozen=True)
class PathRule:
    name: str
    kind: str  # exact | glob | regex
    pattern: str
    rationale: str
    root_only: bool = False

    def matches(self, rel_path: str) -> bool:
        target = rel_path.replace("\\", "/")
        basename = Path(target).name
        candidate = basename if self.root_only else target
        if self.root_only and "/" in target:
            return False
        if self.kind == "exact":
            return candidate == self.pattern
        if self.kind == "glob":
            return fnmatch.fnmatch(candidate, self.pattern)
        if self.kind == "regex":
            return re.fullmatch(self.pattern, candidate) is not None
        raise ValueError(f"Unsupported rule kind: {self.kind}")


@dataclass(frozen=True)
class ContentRule:
    name: str
    pattern: re.Pattern[str]
    rationale: str


@dataclass(frozen=True)
class Violation:
    rel_path: str
    rule_name: str
    detail: str
    rationale: str


PATH_RULES: Sequence[PathRule] = (
    PathRule("DuckDB artifact", "glob", "*.duckdb*", "Database artifacts must not be committed."),
    PathRule("Backup artifact", "glob", "*.bak", "Backup files must not be committed."),
    PathRule("Lock artifact", "glob", "*.lock", "Runtime lock files must not be committed."),
    PathRule("Secret env file", "glob", ".env*", "Environment secret files must not be committed."),
    PathRule("Secret config file", "exact", "config.json", "Secret config files must not be committed."),
    PathRule("Python cache artifact", "glob", "*.pyc", "Compiled Python files must not be committed."),
    PathRule("Log artifact", "glob", "*.log", "Log files must not be committed."),
    PathRule("Patch reject artifact", "glob", "*.rej", "Rejected patch files must not be committed."),
    PathRule("Patch backup artifact", "glob", "*.orig", "Patch backup files must not be committed."),
    PathRule("Patch artifact", "glob", "*.patch", "Patch artifacts must not be committed."),
    PathRule("Diff artifact", "glob", "*.diff", "Diff artifacts must not be committed."),
    PathRule(
        "Timestamped config snapshot",
        "regex",
        r"src/config/config\.\d{8}T\d{6}Z\.yaml",
        "Timestamped config snapshots are generated artifacts and must not be committed.",
    ),
    PathRule(
        "Root patch helper",
        "regex",
        r"patch_.+\.(?:py|sh|txt)",
        "One-off patch helper scripts must not live at repo root.",
        root_only=True,
    ),
    PathRule(
        "Root generated-output placeholder",
        "exact",
        "derived_levels",
        "Generated-output placeholders must not live at repo root.",
        root_only=True,
    ),
    PathRule(
        "Root type-expression temp file",
        "regex",
        r"(?i:(optional|list|dict|tuple|set|literal|union|callable|annotated))\[.*",
        "Accidental type-expression temp files must not live at repo root.",
        root_only=True,
    ),
    PathRule(
        "Root object temp file",
        "regex",
        r"(?i:(uuid\.uuid|duckdb\.duckdbpyconnection))",
        "Accidental object-repr temp files must not live at repo root.",
        root_only=True,
    ),
    PathRule(
        "Root scalar temp file",
        "regex",
        r"(?i:(int|float|bool|str|none))",
        "Accidental scalar temp files must not live at repo root.",
        root_only=True,
    ),
)

CONTENT_RULES: Sequence[ContentRule] = (
    ContentRule(
        "Merge conflict marker",
        re.compile(r"^(<<<<<<<|=======|>>>>>>>)( .*)?$", re.MULTILINE),
        "Unresolved merge conflicts must not remain in committed files.",
    ),
    ContentRule(
        "Apply-patch marker",
        re.compile(r"^\*\*\* (Begin|End) Patch$", re.MULTILINE),
        "Unresolved patch fragments must not remain in committed files.",
    ),
)


def _git_ls_files(root: Path) -> List[str] | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z"],
            check=True,
            capture_output=True,
            text=False,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return [entry for entry in result.stdout.decode("utf-8", errors="replace").split("\0") if entry]


def _walk_repo(root: Path) -> List[str]:
    rel_paths: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for filename in filenames:
            abs_path = Path(dirpath) / filename
            rel_paths.append(abs_path.relative_to(root).as_posix())
    return sorted(rel_paths)


def discover_paths(root: Path) -> List[str]:
    git_files = _git_ls_files(root)
    if git_files is not None:
        return sorted(git_files)
    return _walk_repo(root)


def _should_scan_contents(path: Path) -> bool:
    if path.suffix.lower() in TEXT_SUFFIXES:
        return True
    if path.suffix:
        return False
    return path.stat().st_size <= 1_000_000


def scan_repo(root: Path) -> List[Violation]:
    violations: List[Violation] = []
    for rel_path in discover_paths(root):
        for rule in PATH_RULES:
            if rule.matches(rel_path):
                violations.append(
                    Violation(rel_path=rel_path, rule_name=rule.name, detail=rule.pattern, rationale=rule.rationale)
                )
                break
        else:
            abs_path = root / rel_path
            if not abs_path.is_file() or not _should_scan_contents(abs_path):
                continue
            try:
                text = abs_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for rule in CONTENT_RULES:
                match = rule.pattern.search(text)
                if match is None:
                    continue
                line_no = text[: match.start()].count("\n") + 1
                snippet = match.group(0).strip()[:120]
                violations.append(
                    Violation(
                        rel_path=rel_path,
                        rule_name=rule.name,
                        detail=f"line {line_no}: {snippet}",
                        rationale=rule.rationale,
                    )
                )
                break
    return violations


def format_report(violations: Sequence[Violation]) -> str:
    lines = ["[FAIL] Repo Guard failed. Forbidden artifacts or fragments were found."]
    for violation in violations:
        lines.extend(
            [
                f"- path: {violation.rel_path}",
                f"  rule: {violation.rule_name}",
                f"  detail: {violation.detail}",
                f"  why: {violation.rationale}",
            ]
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fail CI when repo debris or unresolved patch fragments are present.")
    parser.add_argument("--root", default=".", help="Repository root to scan. Defaults to the current directory.")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    violations = scan_repo(root)
    if violations:
        print(format_report(violations))
        return 2
    print(f"[PASS] Repo Guard passed for {root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
