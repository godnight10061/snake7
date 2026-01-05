import numpy as np
import pytest

import gymnasium as gym
from gymnasium.utils.env_checker import check_env


def test_env_checker():
    from snake7.env import SnakeEnv

    env = SnakeEnv(width=6, height=6, max_steps=50)
    check_env(env, skip_render_check=True)


def test_reset_and_step_contract():
    from snake7.env import SnakeEnv

    env = SnakeEnv(width=6, height=6, max_steps=50)
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    assert isinstance(info, dict)

    obs2, reward, terminated, truncated, info2 = env.step(0)
    assert env.observation_space.contains(obs2)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info2, dict)


def test_seed_determinism_first_steps():
    from snake7.env import SnakeEnv

    actions = [0, 2, 0, 1, 0]
    env1 = SnakeEnv(width=6, height=6, max_steps=50)
    env2 = SnakeEnv(width=6, height=6, max_steps=50)

    obs1, _ = env1.reset(seed=123)
    obs2, _ = env2.reset(seed=123)
    assert np.allclose(obs1, obs2)

    for a in actions:
        out1 = env1.step(a)
        out2 = env2.step(a)
        obs1, r1, term1, trunc1, _ = out1
        obs2, r2, term2, trunc2, _ = out2
        assert np.allclose(obs1, obs2)
        assert r1 == r2
        assert term1 == term2
        assert trunc1 == trunc2
        if term1 or trunc1:
            break


def test_rollout_always_ends_by_max_steps():
    from snake7.env import SnakeEnv

    env = SnakeEnv(width=6, height=6, max_steps=20)
    env.reset(seed=0)
    for _ in range(200):
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        if terminated or truncated:
            return
    pytest.fail("Episode did not terminate/truncate within a reasonable bound")
