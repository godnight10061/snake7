from __future__ import annotations

import sys

import pytest


def test_train_passes_device_to_recurrentppo(monkeypatch):
    sb3_contrib = pytest.importorskip("sb3_contrib")

    seen: dict[str, str | None] = {"device": None}

    class DummyRecurrentPPO:
        def __init__(self, *args, **kwargs):
            seen["device"] = kwargs.get("device")

        def learn(self, *args, **kwargs):
            return self

        def save(self, *args, **kwargs):
            return None

    monkeypatch.setattr(sb3_contrib, "RecurrentPPO", DummyRecurrentPPO)

    import snake7.train as train

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "snake7.train",
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

