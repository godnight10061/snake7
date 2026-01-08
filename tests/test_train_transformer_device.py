from __future__ import annotations

import sys

import pytest


def test_train_transformer_passes_device_to_ppo(monkeypatch):
    stable_baselines3 = pytest.importorskip("stable_baselines3")

    seen: dict[str, str | None] = {"device": None}

    class DummyPPO:
        def __init__(self, *args, **kwargs):
            seen["device"] = kwargs.get("device")

        def learn(self, *args, **kwargs):
            return self

        def save(self, *args, **kwargs):
            return None

    monkeypatch.setattr(stable_baselines3, "PPO", DummyPPO)

    import snake7.train_transformer as train

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "snake7.train_transformer",
            "--total-timesteps",
            "1",
            "--n-envs",
            "1",
            "--device",
            "cpu",
        ],
    )

    train.main()
    assert seen["device"] == "cpu"

