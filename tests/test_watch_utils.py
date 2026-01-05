from __future__ import annotations

import os
from pathlib import Path

import pytest


def test_import_watch_module():
    import snake7.watch as watch

    assert hasattr(watch, "main")
    assert hasattr(watch, "resolve_model_path")


def test_resolve_prefers_best_model_zip(tmp_path: Path):
    from snake7.watch import resolve_model_path

    (tmp_path / "other.zip").write_bytes(b"")
    (tmp_path / "snake_ppo_lstm.zip").write_bytes(b"")
    (tmp_path / "best_model.zip").write_bytes(b"")

    assert resolve_model_path(model_path=None, model_dir=tmp_path).name == "best_model.zip"


def test_resolve_falls_back_to_default_name(tmp_path: Path):
    from snake7.watch import resolve_model_path

    (tmp_path / "other.zip").write_bytes(b"")
    (tmp_path / "snake_ppo_lstm.zip").write_bytes(b"")

    assert resolve_model_path(model_path=None, model_dir=tmp_path).name == "snake_ppo_lstm.zip"


def test_resolve_falls_back_to_newest_zip(tmp_path: Path):
    from snake7.watch import resolve_model_path

    a = tmp_path / "a.zip"
    b = tmp_path / "b.zip"
    a.write_bytes(b"")
    b.write_bytes(b"")

    os.utime(a, (1, 1))
    os.utime(b, (2, 2))

    assert resolve_model_path(model_path=None, model_dir=tmp_path).name == "b.zip"


def test_resolve_newest_zip_tie_break_is_deterministic(tmp_path: Path):
    from snake7.watch import resolve_model_path

    b = tmp_path / "b.zip"
    c = tmp_path / "c.zip"
    b.write_bytes(b"")
    c.write_bytes(b"")

    os.utime(b, (2, 2))
    os.utime(c, (2, 2))

    assert resolve_model_path(model_path=None, model_dir=tmp_path).name == "b.zip"


def test_resolve_raises_when_no_models_found(tmp_path: Path):
    from snake7.watch import resolve_model_path

    with pytest.raises(FileNotFoundError):
        resolve_model_path(model_path=None, model_dir=tmp_path)

