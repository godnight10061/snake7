from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from snake7.env import SnakeEnv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Play Snake with a trained PPO+LSTM agent.")
    p.add_argument("--model", type=Path, default=Path("snake_ppo_lstm.zip"))
    p.add_argument("--width", type=int, default=10)
    p.add_argument("--height", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fps", type=float, default=10.0)
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--deterministic", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from sb3_contrib import RecurrentPPO
    except Exception as e:  # pragma: no cover
        raise SystemExit("sb3_contrib is required: pip install sb3-contrib") from e

    model = RecurrentPPO.load(str(args.model))

    env = SnakeEnv(
        width=args.width,
        height=args.height,
        max_steps=args.max_steps,
        render_mode="human",
    )

    delay = 0.0 if args.fps <= 0 else 1.0 / args.fps
    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        lstm_state = None
        episode_start = np.ones((1,), dtype=bool)

        while True:
            action, lstm_state = model.predict(
                obs,
                state=lstm_state,
                episode_start=episode_start,
                deterministic=args.deterministic,
            )
            obs, reward, terminated, truncated, info = env.step(int(action))
            episode_start = np.array([terminated or truncated], dtype=bool)
            if delay:
                time.sleep(delay)
            if terminated or truncated:
                print(f"episode={ep} score={info.get('score')} steps={info.get('steps')}")
                break


if __name__ == "__main__":
    main()

