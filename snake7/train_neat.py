from __future__ import annotations

import argparse
import pickle
import sys
import time
from math import prod
from pathlib import Path
from typing import Any

# Allow running this file directly (e.g. `python snake7/train_neat.py`) by ensuring the
# repository root is on sys.path. Recommended usage is still `python -m snake7.train_neat`.
if __package__ in (None, "") and __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    __package__ = "snake7"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train NEAT agent for Snake.")
    p.add_argument("--width", type=int, default=10)
    p.add_argument("--height", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--max-steps-without-food", type=int, default=None)
    p.add_argument("--step-penalty", type=float, default=-0.01)
    p.add_argument("--truncation-penalty", type=float, default=0.0)
    p.add_argument("--distance-shaping", type=float, default=0.0)
    p.add_argument("--distance-shaping-clip", type=float, default=0.0)
    p.add_argument("--food-reward", type=float, default=1.0)
    p.add_argument("--death-penalty", type=float, default=-1.0)
    p.add_argument("--win-reward", type=float, default=5.0)

    p.add_argument("--pop-size", type=int, default=50)
    p.add_argument("--generations", type=int, default=None, help="If omitted, evolve until early stop.")
    p.add_argument("--episodes-per-genome", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--fitness-reward-weight", type=float, default=1.0)
    p.add_argument("--fitness-score-weight", type=float, default=0.0)
    p.add_argument("--fitness-steps-weight", type=float, default=0.0)
    p.add_argument("--fitness-agg", choices=["mean", "max"], default="mean")

    p.add_argument("--patience-seconds", type=float, default=300.0)
    p.add_argument("--min-delta", type=float, default=0.01)

    p.add_argument("--genome-out", type=Path, default=Path("snake_neat_genome.pkl"))
    p.add_argument("--config-out", type=Path, default=Path("neat_config.txt"))

    return p.parse_args()


NEAT_CONFIG_TEMPLATE = """
[NEAT]
fitness_criterion     = max
fitness_threshold     = 1000000
pop_size              = {pop_size}
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
num_inputs              = {num_inputs}
num_outputs             = {num_outputs}
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
"""


def render_neat_config(*, pop_size: int, num_inputs: int, num_outputs: int) -> str:
    return (
        NEAT_CONFIG_TEMPLATE.format(
            pop_size=int(pop_size),
            num_inputs=int(num_inputs),
            num_outputs=int(num_outputs),
        ).lstrip()
        + "\n"
    )


def select_action(outputs: Any) -> int:
    """
    Convert NEAT network outputs to a discrete action.

    Snake action space: 0=straight, 1=left, 2=right.
    """
    return max(range(len(outputs)), key=lambda i: float(outputs[i]))


def eval_genome(
    genome: Any,
    config: Any,
    *,
    env_kwargs: dict[str, Any],
    episodes: int,
    seed: int,
    fitness_reward_weight: float = 1.0,
    fitness_score_weight: float = 0.0,
    fitness_steps_weight: float = 0.0,
    fitness_agg: str = "mean",
) -> float:
    import neat

    from snake7.env import SnakeEnv

    net = neat.nn.FeedForwardNetwork.create(genome, config)
    fitness_reward_weight = float(fitness_reward_weight)
    fitness_score_weight = float(fitness_score_weight)
    fitness_steps_weight = float(fitness_steps_weight)

    env = SnakeEnv(**env_kwargs)
    try:
        episode_fitnesses: list[float] = []
        for i in range(int(episodes)):
            obs, _ = env.reset(seed=int(seed) + i)
            episode_reward = 0.0
            last_info: dict[str, Any] = {}
            while True:
                action = select_action(net.activate(obs.tolist()))
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += float(reward)
                last_info = info
                if terminated or truncated:
                    break

            episode_score = float(last_info.get("score", 0.0))
            episode_steps = float(last_info.get("steps", 0.0))
            episode_fitness = (
                fitness_reward_weight * episode_reward
                + fitness_score_weight * episode_score
                + fitness_steps_weight * episode_steps
            )
            episode_fitnesses.append(float(episode_fitness))
    finally:
        env.close()

    if not episode_fitnesses:
        return float("-inf")

    if fitness_agg == "max":
        return max(episode_fitnesses)
    if fitness_agg == "mean":
        return sum(episode_fitnesses) / float(len(episode_fitnesses))
    raise ValueError(f"Unsupported fitness_agg: {fitness_agg}")


def eval_genomes(
    genomes: Any,
    config: Any,
    *,
    env_kwargs: dict[str, Any],
    episodes_per_genome: int,
    seed: int,
    fitness_reward_weight: float = 1.0,
    fitness_score_weight: float = 0.0,
    fitness_steps_weight: float = 0.0,
    fitness_agg: str = "mean",
) -> float:
    best = float("-inf")
    for _, genome in genomes:
        fitness = eval_genome(
            genome,
            config,
            env_kwargs=env_kwargs,
            episodes=int(episodes_per_genome),
            seed=int(seed),
            fitness_reward_weight=float(fitness_reward_weight),
            fitness_score_weight=float(fitness_score_weight),
            fitness_steps_weight=float(fitness_steps_weight),
            fitness_agg=str(fitness_agg),
        )
        genome.fitness = fitness
        if fitness > best:
            best = fitness
    return best


def main() -> None:
    args = parse_args()

    try:
        import neat
    except ImportError:
        print("neat-python is required: pip install neat-python")
        sys.exit(1)

    from snake7.env import SnakeEnv

    env_kwargs = {
        "width": args.width,
        "height": args.height,
        "max_steps": args.max_steps,
        "max_steps_without_food": args.max_steps_without_food,
        "step_penalty": args.step_penalty,
        "truncation_penalty": args.truncation_penalty,
        "distance_shaping": args.distance_shaping,
        "distance_shaping_clip": args.distance_shaping_clip,
        "food_reward": args.food_reward,
        "death_penalty": args.death_penalty,
        "win_reward": args.win_reward,
    }

    # Determine inputs/outputs from environment
    temp_env = SnakeEnv(**env_kwargs)
    num_inputs = int(prod(temp_env.observation_space.shape))
    num_outputs = int(temp_env.action_space.n)
    temp_env.close()

    # Build config
    config_content = render_neat_config(
        pop_size=args.pop_size,
        num_inputs=num_inputs,
        num_outputs=num_outputs,
    )
    args.config_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.config_out, "w") as f:
        f.write(config_content)

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(args.config_out),
    )

    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    class EarlyStopException(Exception):
        pass

    class WallTimeEarlyStopReporter(neat.reporting.BaseReporter):
        def __init__(self, patience_seconds: float, min_delta: float):
            self.patience_seconds = patience_seconds
            self.min_delta = min_delta
            self.best_fitness = float("-inf")
            self.last_improvement_time = time.monotonic()

        def post_evaluate(self, _config, _population, _species, best_genome):
            now = time.monotonic()
            if best_genome.fitness > self.best_fitness + self.min_delta:
                self.best_fitness = best_genome.fitness
                self.last_improvement_time = now
            elif now - self.last_improvement_time > self.patience_seconds:
                print(f"\nEarly stopping: no improvement for {now - self.last_improvement_time:.1f}s")
                raise EarlyStopException()

    early_stop = WallTimeEarlyStopReporter(args.patience_seconds, args.min_delta)
    p.add_reporter(early_stop)

    def _eval_genomes(genomes, config):
        eval_genomes(
            genomes,
            config,
            env_kwargs=env_kwargs,
            episodes_per_genome=int(args.episodes_per_genome),
            seed=int(args.seed),
            fitness_reward_weight=float(args.fitness_reward_weight),
            fitness_score_weight=float(args.fitness_score_weight),
            fitness_steps_weight=float(args.fitness_steps_weight),
            fitness_agg=str(args.fitness_agg),
        )

    print(f"\nStarting training for {args.generations if args.generations else 'infinite'} generations.")
    print(f"Early stop patience: {args.patience_seconds:.1f}s (min delta: {args.min_delta})")
    print(
        "Fitness weights:"
        f" reward={float(args.fitness_reward_weight)}"
        f" score={float(args.fitness_score_weight)}"
        f" steps={float(args.fitness_steps_weight)}"
        f" agg={str(args.fitness_agg)}"
    )

    try:
        winner = p.run(_eval_genomes, args.generations)
        print("\nTraining finished (generations limit reached or fitness threshold met).")
    except EarlyStopException:
        print(f"\nTraining stopped due to patience ({args.patience_seconds}s limit reached).")
        winner = stats.best_genome()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        winner = stats.best_genome()

    if winner is None:
        print("\nNo genome evolved (interrupted early?).")
        return

    # Save best genome
    args.genome_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.genome_out, "wb") as f:
        pickle.dump(winner, f)

    print(f"\nBest fitness: {winner.fitness}")
    print(f"Best genome saved to {args.genome_out}")
    print(f"Config saved to {args.config_out}")


if __name__ == "__main__":
    main()
