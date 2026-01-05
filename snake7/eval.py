from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from snake7.watch import resolve_model_path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained Snake agent and report mean episode reward.")
    p.add_argument("--model", type=Path, default=None, help="Path to model .zip (optional).")
    p.add_argument("--model-dir", type=Path, default=Path("."), help="Directory to search for model files.")

    p.add_argument("--width", type=int, default=10)
    p.add_argument("--height", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--step-penalty", type=float, default=-0.01)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--deterministic", action="store_true")

    p.add_argument(
        "--require-mean-reward",
        type=float,
        default=None,
        help="Exit non-zero unless mean_reward exceeds this threshold.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    try:
        from sb3_contrib import RecurrentPPO
    except Exception as e:  # pragma: no cover
        raise SystemExit("sb3_contrib is required: pip install sb3-contrib") from e

    from snake7.env import SnakeEnv

    if args.episodes <= 0:
        raise SystemExit("--episodes must be > 0")

    model_path = resolve_model_path(model_path=args.model, model_dir=args.model_dir)
    model = RecurrentPPO.load(str(model_path), device="cpu")

    rewards: list[float] = []
    scores: list[int] = []

    env = SnakeEnv(
        width=args.width,
        height=args.height,
        max_steps=args.max_steps,
        step_penalty=args.step_penalty,
        render_mode=None,
    )

    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        lstm_state = None
        episode_start = np.ones((1,), dtype=bool)
        ep_reward = 0.0

        while True:
            action, lstm_state = model.predict(
                obs,
                state=lstm_state,
                episode_start=episode_start,
                deterministic=args.deterministic,
            )
            obs, reward, terminated, truncated, info = env.step(int(action))
            ep_reward += float(reward)
            episode_start = np.array([terminated or truncated], dtype=bool)
            if terminated or truncated:
                break

        rewards.append(ep_reward)
        scores.append(int(info.get("score", 0)))

    mean_reward = float(np.mean(rewards))
    std_reward = float(np.std(rewards))
    mean_score = float(np.mean(scores))

    print(f"model={model_path}")
    print(f"episodes={args.episodes} mean_reward={mean_reward:.4f} std_reward={std_reward:.4f} mean_score={mean_score:.3f}")

    if args.require_mean_reward is not None and not (mean_reward > float(args.require_mean_reward)):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
