import pytest


def test_recurrentppo_learn_smoke():
    sb3_contrib = pytest.importorskip("sb3_contrib")
    pytest.importorskip("stable_baselines3")

    from stable_baselines3.common.monitor import Monitor

    from snake7.env import SnakeEnv

    env = Monitor(SnakeEnv(width=6, height=6, max_steps=50))
    model = sb3_contrib.RecurrentPPO(
        "MlpLstmPolicy",
        env,
        n_steps=16,
        batch_size=16,
        seed=0,
        verbose=0,
    )
    model.learn(total_timesteps=32)

    obs, _ = env.reset(seed=0)
    action, _ = model.predict(obs, deterministic=True)
    assert env.action_space.contains(int(action))

