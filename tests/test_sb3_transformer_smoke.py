import pytest


def test_ppo_transformer_learn_smoke():
    pytest.importorskip("stable_baselines3")

    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor

    from snake7.env import SnakeEnv
    from snake7.transformer import TransformerFeaturesExtractor
    from snake7.wrappers import ObsStackWrapper

    env = Monitor(ObsStackWrapper(SnakeEnv(width=6, height=6, max_steps=50), n_stack=4))
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=16,
        batch_size=16,
        device="cpu",
        seed=0,
        verbose=0,
        policy_kwargs={
            "features_extractor_class": TransformerFeaturesExtractor,
            "features_extractor_kwargs": {"d_model": 32, "n_head": 4, "n_layers": 1},
        },
    )
    model.learn(total_timesteps=32)

    obs, _ = env.reset(seed=0)
    action, _ = model.predict(obs, deterministic=True)
    assert env.action_space.contains(int(action))
