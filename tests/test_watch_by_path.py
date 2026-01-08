from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_watch_can_run_by_path(tmp_path: Path):
    pytest.importorskip("sb3_contrib")
    pytest.importorskip("stable_baselines3")

    from sb3_contrib import RecurrentPPO

    from snake7.env import SnakeEnv

    env = SnakeEnv(width=6, height=6, max_steps=10, step_penalty=0)
    model = RecurrentPPO("MlpLstmPolicy", env, device="cpu", n_steps=16, batch_size=16, verbose=0)
    model_path = tmp_path / "model.zip"
    model.save(str(model_path))

    repo_root = Path(__file__).resolve().parents[1]

    env_vars = os.environ.copy()
    env_vars.pop("PYTHONPATH", None)
    env_vars.pop("PYTHONHOME", None)

    proc = subprocess.run(
        [
            sys.executable,
            str(repo_root / "snake7" / "watch.py"),
            "--model",
            str(model_path),
            "--width",
            "6",
            "--height",
            "6",
            "--max-steps",
            "1",
            "--episodes",
            "1",
            "--fps",
            "0",
            "--no-ansi",
        ],
        cwd=str(repo_root),
        env=env_vars,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert "No module named 'snake7'" not in proc.stderr


def test_watch_can_run_by_path_for_transformer_ppo(tmp_path: Path):
    pytest.importorskip("stable_baselines3")

    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor

    from snake7.env import SnakeEnv
    from snake7.transformer import TransformerFeaturesExtractor
    from snake7.wrappers import ObsStackWrapper

    env = Monitor(ObsStackWrapper(SnakeEnv(width=6, height=6, max_steps=10, step_penalty=0), n_stack=4))
    model = PPO(
        "MlpPolicy",
        env,
        device="cpu",
        n_steps=16,
        batch_size=16,
        seed=0,
        verbose=0,
        policy_kwargs={
            "features_extractor_class": TransformerFeaturesExtractor,
            "features_extractor_kwargs": {"d_model": 32, "n_head": 4, "n_layers": 1},
        },
    )
    model.learn(total_timesteps=32)

    model_path = tmp_path / "model.zip"
    model.save(str(model_path))

    repo_root = Path(__file__).resolve().parents[1]

    env_vars = os.environ.copy()
    env_vars.pop("PYTHONPATH", None)
    env_vars.pop("PYTHONHOME", None)

    proc = subprocess.run(
        [
            sys.executable,
            str(repo_root / "snake7" / "watch.py"),
            "--model",
            str(model_path),
            "--width",
            "6",
            "--height",
            "6",
            "--max-steps",
            "1",
            "--episodes",
            "1",
            "--fps",
            "0",
            "--no-ansi",
        ],
        cwd=str(repo_root),
        env=env_vars,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert "No module named 'snake7'" not in proc.stderr
