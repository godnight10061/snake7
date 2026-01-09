from __future__ import annotations

import builtins
import importlib
import sys
from pathlib import Path

import pytest


def _make_blocking_import(*, blocked_prefixes: tuple[str, ...]):
    real_import = builtins.__import__

    def _blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
        for prefix in blocked_prefixes:
            if name == prefix or name.startswith(prefix + "."):
                raise ImportError(f"blocked import: {name}")
        return real_import(name, globals, locals, fromlist, level)

    return _blocked_import


def test_watch_import_does_not_require_sb3(monkeypatch):
    """
    Importing `snake7.watch` should work even if SB3 isn't installed,
    as long as we aren't loading a `.zip` model.
    """
    monkeypatch.setattr(
        builtins,
        "__import__",
        _make_blocking_import(blocked_prefixes=("stable_baselines3", "sb3_contrib")),
    )
    sys.modules.pop("snake7.watch", None)
    importlib.import_module("snake7.watch")


def test_watch_neat_missing_dependency_message(tmp_path: Path, monkeypatch):
    """
    When watching a `.pkl` NEAT genome without neat-python, show a copy/pastable install hint.
    """
    import snake7.watch as watch

    model_path = tmp_path / "best_genome.pkl"
    model_path.write_bytes(b"not a real pickle")

    monkeypatch.setattr(
        builtins,
        "__import__",
        _make_blocking_import(blocked_prefixes=("neat",)),
    )
    monkeypatch.setattr(
        watch,
        "_parse_args",
        lambda: type(
            "Args",
            (),
            {
                "model": model_path,
                "model_dir": tmp_path,
                "width": 6,
                "height": 6,
                "max_steps": 1,
                "seed": 0,
                "episodes": 1,
                "fps": 0.0,
                "deterministic": True,
                "no_ansi": True,
            },
        )(),
    )

    with pytest.raises(SystemExit) as excinfo:
        watch.main()

    msg = str(excinfo.value)
    assert "neat-python is required for .pkl models" in msg
    assert "python -m pip install neat-python" in msg
