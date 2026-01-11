from __future__ import annotations

from unittest.mock import MagicMock

import pytest


def test_eval_genome_weighted_score_and_agg(monkeypatch):
    neat = pytest.importorskip("neat")

    import numpy as np

    import snake7.env as env_mod
    import snake7.train_neat as train_neat

    class _Net:
        def activate(self, obs_list):
            return [0.0, 0.0, 0.0]

    monkeypatch.setattr(neat.nn.FeedForwardNetwork, "create", lambda _g, _c: _Net())

    class _FakeEnv:
        def __init__(self, **_kwargs):
            self._seed = 0

        def reset(self, *, seed: int):
            self._seed = int(seed)
            return np.zeros((9,), dtype=np.float32), {}

        def step(self, _action: int):
            # Two episodes with different (reward, score) so we can test agg.
            if self._seed % 2 == 0:
                reward = 10.0
                score = 3
            else:
                reward = 1.0
                score = 0
            info = {"score": score, "steps": 1}
            return np.zeros((9,), dtype=np.float32), float(reward), True, False, info

        def close(self):
            return None

    monkeypatch.setattr(env_mod, "SnakeEnv", _FakeEnv)

    # Per-episode fitness = reward_weight*reward + score_weight*score
    # seed 0 => 10 + 100*3 = 310
    # seed 1 =>  1 + 100*0 =   1
    # mean => 155.5, max => 310
    genome = MagicMock()
    config = MagicMock()

    mean_fit = train_neat.eval_genome(
        genome,
        config,
        env_kwargs={},
        episodes=2,
        seed=0,
        fitness_reward_weight=1.0,
        fitness_score_weight=100.0,
        fitness_agg="mean",
    )
    assert mean_fit == pytest.approx(155.5)

    max_fit = train_neat.eval_genome(
        genome,
        config,
        env_kwargs={},
        episodes=2,
        seed=0,
        fitness_reward_weight=1.0,
        fitness_score_weight=100.0,
        fitness_agg="max",
    )
    assert max_fit == pytest.approx(310.0)

