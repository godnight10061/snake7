from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _clean_env() -> dict[str, str]:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    return env


def _run_help(repo_root: Path, rel_path: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(repo_root / rel_path), "--help"],
        cwd=str(repo_root),
        env=_clean_env(),
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_train_can_run_by_path_showing_help():
    repo_root = Path(__file__).resolve().parents[1]
    proc = _run_help(repo_root, "snake7/train.py")
    assert proc.returncode == 0, proc.stderr
    assert "ModuleNotFoundError" not in proc.stderr
    assert "No module named 'snake7'" not in proc.stderr


def test_eval_can_run_by_path_showing_help():
    repo_root = Path(__file__).resolve().parents[1]
    proc = _run_help(repo_root, "snake7/eval.py")
    assert proc.returncode == 0, proc.stderr
    assert "ModuleNotFoundError" not in proc.stderr
    assert "No module named 'snake7'" not in proc.stderr


def test_play_can_run_by_path_showing_help():
    repo_root = Path(__file__).resolve().parents[1]
    proc = _run_help(repo_root, "snake7/play.py")
    assert proc.returncode == 0, proc.stderr
    assert "ModuleNotFoundError" not in proc.stderr
    assert "No module named 'snake7'" not in proc.stderr

