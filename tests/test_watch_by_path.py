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


def test_watch_can_run_neat_by_path(tmp_path: Path):
    neat = pytest.importorskip("neat")

    import pickle
    import random

    import numpy as np

    from snake7.env import SnakeEnv

    env = SnakeEnv()
    num_inputs = env.observation_space.shape[0]
    num_outputs = env.action_space.n

    cfg_path = tmp_path / "neat_config.txt"
    cfg_path.write_text(
        (
            f"""
[NEAT]
fitness_criterion     = max
fitness_threshold     = 1000000
pop_size              = 2
reset_on_extinction   = False
no_fitness_termination = False

[DefaultGenome]
# node activation options
activation_default      = tanh
activation_mutate_rate  = 0.0
activation_options      = tanh

# node aggregation options
aggregation_default     = sum
aggregation_mutate_rate = 0.0
aggregation_options     = sum

# node bias options
bias_init_type          = gaussian
bias_init_mean          = 0.0
bias_init_stdev         = 1.0
bias_max_value          = 30.0
bias_min_value          = -30.0
bias_mutate_power       = 0.5
bias_mutate_rate        = 0.7
bias_replace_rate       = 0.1

# node response options
response_init_type      = gaussian
response_init_mean      = 1.0
response_init_stdev     = 0.0
response_max_value      = 30.0
response_min_value      = -30.0
response_mutate_power   = 0.0
response_mutate_rate    = 0.0
response_replace_rate   = 0.0

# genome compatibility options
compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient   = 0.5

# connection add/remove rates
conn_add_prob           = 0.5
conn_delete_prob        = 0.5

# connection enable options
enabled_default         = True
enabled_mutate_rate     = 0.01
enabled_rate_to_true_add  = 0.0
enabled_rate_to_false_add = 0.0

# connection weight options
weight_init_type        = gaussian
weight_init_mean        = 0.0
weight_init_stdev       = 1.0
weight_max_value        = 30
weight_min_value        = -30
weight_mutate_power     = 0.5
weight_mutate_rate        = 0.8
weight_replace_rate     = 0.1

# node add/remove rates
node_add_prob           = 0.2
node_delete_prob        = 0.2

# network parameters
num_hidden              = 0
num_inputs              = {int(num_inputs)}
num_outputs             = {int(num_outputs)}
feed_forward            = True
initial_connection      = full

# structural mutation options
single_structural_mutation = False
structural_mutation_surer   = default

[DefaultSpeciesSet]
compatibility_threshold = 3.0

[DefaultStagnation]
species_fitness_func = max
max_stagnation       = 20
species_elitism      = 2

[DefaultReproduction]
elitism            = 2
survival_threshold = 0.2
min_species_size   = 2
""".lstrip()
            + "\n"
        ),
        encoding="utf-8",
    )

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(cfg_path),
    )

    random.seed(0)
    np.random.seed(0)
    pop = neat.Population(config)
    genome = pop.population[min(pop.population.keys())]

    genome_path = tmp_path / "best_genome.pkl"
    with open(genome_path, "wb") as f:
        pickle.dump(genome, f)

    repo_root = Path(__file__).resolve().parents[1]

    env_vars = os.environ.copy()
    env_vars.pop("PYTHONPATH", None)
    env_vars.pop("PYTHONHOME", None)

    proc = subprocess.run(
        [
            sys.executable,
            str(repo_root / "snake7" / "watch.py"),
            "--model",
            str(genome_path),
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
