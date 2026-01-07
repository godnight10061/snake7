from __future__ import annotations

import pytest


def test_hunger_truncation_triggers():
    from snake7.env import SnakeEnv

    env = SnakeEnv(width=6, height=6, max_steps=100, step_penalty=0.0, max_steps_without_food=3, truncation_penalty=-1.0)
    env.reset(seed=0, options={"food": (0, 0)})  # keep food away from the first few moves

    actions = [1, 1, 1]  # turn left 3 times => safe square-ish path
    last = None
    for a in actions:
        last = env.step(a)
    assert last is not None
    _, reward, terminated, truncated, _ = last
    assert terminated is False
    assert truncated is True
    assert reward == pytest.approx(-1.0)


def test_distance_shaping_rewards_moving_toward_food():
    from snake7.env import SnakeEnv

    env = SnakeEnv(
        width=6,
        height=6,
        max_steps=100,
        step_penalty=0.0,
        distance_shaping=0.1,
        distance_shaping_clip=0.05,
    )
    env.reset(
        seed=0,
        options={
            "snake": [(2, 2), (1, 2), (0, 2)],
            "direction": 1,  # right
            "food": (4, 2),
        },
    )

    _, reward, terminated, truncated, _ = env.step(0)  # straight => closer
    assert terminated is False
    assert truncated is False
    assert reward == pytest.approx(0.05)
