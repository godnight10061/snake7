from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running this file directly (e.g. `python snake7/train_transformer.py`) by ensuring the
# repository root is on sys.path. Recommended usage is still `python -m snake7.train_transformer`.
if __package__ in (None, "") and __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    __package__ = "snake7"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train PPO (Transformer features) agent for Snake (stable-baselines3.PPO).")
    p.add_argument("--width", type=int, default=10)
    p.add_argument("--height", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--max-steps-without-food", type=int, default=None)
    p.add_argument("--step-penalty", type=float, default=-0.001)
    p.add_argument("--truncation-penalty", type=float, default=-1.0)
    p.add_argument("--distance-shaping", type=float, default=0.01)
    p.add_argument("--distance-shaping-clip", type=float, default=0.05)
    p.add_argument("--food-reward", type=float, default=2.0)
    p.add_argument("--death-penalty", type=float, default=-1.0)
    p.add_argument("--win-reward", type=float, default=5.0)

    p.add_argument("--n-stack", type=int, default=8, help="How many last observations to stack (sequence length).")

    p.add_argument("--d-model", type=int, default=64, help="Transformer model dimension.")
    p.add_argument("--n-head", type=int, default=4, help="Transformer attention heads.")
    p.add_argument("--n-layers", type=int, default=2, help="Transformer encoder layers.")
    p.add_argument("--dropout", type=float, default=0.0, help="Transformer dropout.")
    p.add_argument("--pooling", type=str, default="last", choices=["last", "mean"])

    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--total-timesteps", type=int, default=None, help="If omitted, train until early stop/Ctrl+C.")
    p.add_argument("--chunk-timesteps", type=int, default=None, help="Only used for infinite training.")

    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--n-steps", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Torch device for SB3 (e.g., auto, cpu, cuda, cuda:0).",
    )

    p.add_argument("--patience-seconds", type=float, default=300.0)
    p.add_argument("--early-stop-window", type=int, default=20)
    p.add_argument("--early-stop-min-delta", type=float, default=0.0)
    p.add_argument("--stop-mean-reward", type=float, default=None, help="Stop when rolling mean reward exceeds this.")

    p.add_argument("--model-out", type=Path, default=Path("snake_ppo_transformer.zip"))
    p.add_argument("--best-model-out", type=Path, default=None)
    p.add_argument("--tensorboard-log", type=Path, default=None)

    p.add_argument("--eval-freq", type=int, default=50_000, help="Eval frequency in timesteps (0 disables).")
    p.add_argument("--eval-episodes", type=int, default=20)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from stable_baselines3 import PPO
    except Exception as e:  # pragma: no cover
        raise SystemExit("stable-baselines3 is required: pip install stable-baselines3") from e

    from stable_baselines3.common.env_util import make_vec_env

    from snake7.callbacks import WallTimeEarlyStopCallback
    from snake7.env import SnakeEnv
    from snake7.transformer import TransformerFeaturesExtractor
    from snake7.wrappers import ObsStackWrapper

    if args.n_envs <= 0:
        raise SystemExit("--n-envs must be > 0")
    if args.n_steps <= 0:
        raise SystemExit("--n-steps must be > 0")
    if args.n_stack <= 0:
        raise SystemExit("--n-stack must be > 0")
    if args.d_model <= 0:
        raise SystemExit("--d-model must be > 0")
    if args.n_head <= 0:
        raise SystemExit("--n-head must be > 0")
    if args.n_layers <= 0:
        raise SystemExit("--n-layers must be > 0")
    if args.d_model % args.n_head != 0:
        raise SystemExit("--d-model must be divisible by --n-head")
    if args.dropout < 0:
        raise SystemExit("--dropout must be >= 0")
    if args.patience_seconds < 0:
        raise SystemExit("--patience-seconds must be >= 0")
    if args.early_stop_window <= 0:
        raise SystemExit("--early-stop-window must be > 0")
    if args.early_stop_min_delta < 0:
        raise SystemExit("--early-stop-min-delta must be >= 0")
    if args.eval_freq < 0:
        raise SystemExit("--eval-freq must be >= 0")
    if args.eval_episodes <= 0:
        raise SystemExit("--eval-episodes must be > 0")

    total_batch = args.n_envs * args.n_steps
    batch_size = args.batch_size
    if total_batch % batch_size != 0:
        raise SystemExit(f"--batch-size must divide n_envs*n_steps ({total_batch})")

    max_steps_without_food = args.max_steps_without_food
    if max_steps_without_food is None:
        max_steps_without_food = args.width * args.height

    env_kwargs = {
        "width": args.width,
        "height": args.height,
        "max_steps": args.max_steps,
        "max_steps_without_food": max_steps_without_food,
        "step_penalty": args.step_penalty,
        "truncation_penalty": args.truncation_penalty,
        "distance_shaping": args.distance_shaping,
        "distance_shaping_clip": args.distance_shaping_clip,
        "food_reward": args.food_reward,
        "death_penalty": args.death_penalty,
        "win_reward": args.win_reward,
    }

    env = make_vec_env(
        SnakeEnv,
        n_envs=args.n_envs,
        seed=args.seed,
        env_kwargs=env_kwargs,
        wrapper_class=ObsStackWrapper,
        wrapper_kwargs={"n_stack": args.n_stack},
    )

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        ent_coef=args.ent_coef,
        n_steps=args.n_steps,
        batch_size=batch_size,
        device=args.device,
        seed=args.seed,
        verbose=1,
        tensorboard_log=str(args.tensorboard_log) if args.tensorboard_log else None,
        policy_kwargs={
            "features_extractor_class": TransformerFeaturesExtractor,
            "features_extractor_kwargs": {
                "d_model": args.d_model,
                "n_head": args.n_head,
                "n_layers": args.n_layers,
                "dropout": args.dropout,
                "pooling": args.pooling,
            },
        },
    )

    best_model_path = args.best_model_out or (args.model_out.parent / "best_model.zip")
    early_stop_cb = WallTimeEarlyStopCallback(
        patience_seconds=args.patience_seconds,
        window_size=args.early_stop_window,
        min_delta=args.early_stop_min_delta,
        save_best_path=None,
        stop_score_threshold=args.stop_mean_reward,
        verbose=1,
    )

    callbacks = [early_stop_cb]
    if args.eval_freq:
        from stable_baselines3.common.callbacks import EvalCallback

        eval_env = make_vec_env(
            SnakeEnv,
            n_envs=1,
            seed=args.seed + 10_000,
            env_kwargs=env_kwargs,
            wrapper_class=ObsStackWrapper,
            wrapper_kwargs={"n_stack": args.n_stack},
        )

        eval_freq_steps = max(args.eval_freq // args.n_envs, 1)
        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path=str(best_model_path.parent),
            log_path=str(best_model_path.parent),
            eval_freq=eval_freq_steps,
            n_eval_episodes=args.eval_episodes,
            deterministic=True,
        )
        callbacks.append(eval_cb)

    try:
        if args.total_timesteps is not None:
            if args.total_timesteps <= 0:
                raise SystemExit("--total-timesteps must be > 0 (or omit it for infinite training)")
            model.learn(total_timesteps=args.total_timesteps, callback=callbacks)
        else:
            chunk = args.chunk_timesteps
            if chunk is None:
                chunk = max(total_batch * 10, 10_000)
            if chunk <= 0:
                raise SystemExit("--chunk-timesteps must be > 0")
            while True:
                model.learn(total_timesteps=chunk, reset_num_timesteps=False, callback=callbacks)
                if early_stop_cb.stopped:
                    break
    except KeyboardInterrupt:
        pass
    finally:
        args.model_out.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(args.model_out))
        if args.eval_freq:
            best_model_src = best_model_path.parent / "best_model.zip"
            if best_model_src.exists() and best_model_src != best_model_path:
                best_model_path.write_bytes(best_model_src.read_bytes())


if __name__ == "__main__":
    main()

