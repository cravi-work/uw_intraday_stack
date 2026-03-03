from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "repo_guard.py"


def _run_guard(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(root)],
        check=False,
        capture_output=True,
        text=True,
    )


def test_repo_guard_passes_on_clean_tree(tmp_path: Path):
    (tmp_path / "README.md").write_text("clean\n", encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("print('ok')\n", encoding="utf-8")

    result = _run_guard(tmp_path)

    assert result.returncode == 0
    assert "[PASS] Repo Guard passed" in result.stdout


def test_repo_guard_fails_on_root_type_expression_artifact(tmp_path: Path):
    (tmp_path / "Optional[str]").write_text("", encoding="utf-8")

    result = _run_guard(tmp_path)

    assert result.returncode == 2
    assert "Root type-expression temp file" in result.stdout
    assert "Optional[str]" in result.stdout


def test_repo_guard_fails_on_root_patch_helper(tmp_path: Path):
    (tmp_path / "patch_tests.py").write_text("print('oops')\n", encoding="utf-8")

    result = _run_guard(tmp_path)

    assert result.returncode == 2
    assert "Root patch helper" in result.stdout
    assert "patch_tests.py" in result.stdout


def test_repo_guard_fails_on_timestamped_config_snapshot(tmp_path: Path):
    snapshot = tmp_path / "src" / "config"
    snapshot.mkdir(parents=True)
    (snapshot / "config.20260218T133202Z.yaml").write_text("x: 1\n", encoding="utf-8")

    result = _run_guard(tmp_path)

    assert result.returncode == 2
    assert "Timestamped config snapshot" in result.stdout
    assert "src/config/config.20260218T133202Z.yaml" in result.stdout


def test_repo_guard_fails_on_merge_conflict_marker(tmp_path: Path):
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "broken.py").write_text(
        "def f():\n<<<<<<< HEAD\n    return 1\n=======\n    return 2\n>>>>>>> branch\n",
        encoding="utf-8",
    )

    result = _run_guard(tmp_path)

    assert result.returncode == 2
    assert "Merge conflict marker" in result.stdout
    assert "src/broken.py" in result.stdout
