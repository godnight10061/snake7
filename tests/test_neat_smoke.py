from __future__ import annotations

import random

import numpy as np
import pytest


def test_neat_eval_assigns_fitness_and_action_valid(tmp_path):
    neat = pytest.importorskip("neat")

    from snake7.env import SnakeEnv

    import snake7.train_neat as train_neat

    cfg_path = tmp_path / "neat.cfg"
    cfg_text = train_neat.render_neat_config(
        pop_size=2,
        num_inputs=SnakeEnv().observation_space.shape[0],
        num_outputs=SnakeEnv().action_space.n,
    )
    assert "no_fitness_termination" in cfg_text
    cfg_path.write_text(cfg_text, encoding="utf-8")

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(cfg_path),
    )

    random.seed(0)
    np.random.seed(0)

    # Use neat.Population to create initial genomes, as newer neat-python versions
    # require an innovation tracker to be set on the genome config.
    pop = neat.Population(config)
    genomes = list(pop.population.items())

    best_fitness = train_neat.eval_genomes(
        genomes,
        config,
        env_kwargs={"width": 6, "height": 6, "max_steps": 20, "step_penalty": 0.0},
        episodes_per_genome=1,
        seed=0,
    )

    assert isinstance(best_fitness, float)
    assert np.isfinite(best_fitness)
    assert genomes[0][1].fitness is not None
    assert genomes[1][1].fitness is not None

    net = neat.nn.FeedForwardNetwork.create(genomes[0][1], config)
    env = SnakeEnv(width=6, height=6, max_steps=5, step_penalty=0.0)
    obs, _ = env.reset(seed=0)
    action = train_neat.select_action(net.activate(obs.tolist()))
    assert env.action_space.contains(action)
