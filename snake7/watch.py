from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional, Union

# Allow running this file directly (e.g. `python snake7/watch.py`) by ensuring the
# repository root is on sys.path. Recommended usage is still `python -m snake7.watch`.
if __package__ in (None, "") and __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    __package__ = "snake7"


def resolve_model_path(*, model_path: Optional[Union[str, Path]], model_dir: Path) -> Path:
    """
    Resolve which model to load.

    Priority:
    1) Explicit `model_path`
    2) `best_model.zip` in `model_dir`
    3) `snake_ppo_lstm.zip` in `model_dir`
    4) `best_genome.pkl` in `model_dir`
    5) Newest `*.zip` or `*.pkl` in `model_dir` (mtime, tie-break by path)
    """
    model_dir = Path(model_dir)

    if model_path is not None:
        p = Path(model_path)
        if not p.exists():
            raise FileNotFoundError(f"Model not found: {p}")
        return p

    preferred = [
        model_dir / "best_model.zip",
        model_dir / "snake_ppo_lstm.zip",
        model_dir / "best_genome.pkl",
    ]
    for p in preferred:
        if p.exists():
            return p

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    files = sorted(list(model_dir.glob("*.zip")) + list(model_dir.glob("*.pkl")))
    if not files:
        raise FileNotFoundError(f"No .zip or .pkl models found in: {model_dir}")

    def sort_key(p: Path) -> tuple[float, str]:
        try:
            mtime = float(p.stat().st_mtime)
        except OSError:
            mtime = float("-inf")
        # newest first; tie-break by path (ascending) for determinism
        return (-mtime, str(p))

    return sorted(files, key=sort_key)[0]


def _enable_windows_vt_mode() -> None:
    if os.name != "nt":
        return
    try:
        import ctypes  # noqa: WPS433

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE = -11
        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)) == 0:
            return
        ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
        new_mode = mode.value | ENABLE_VIRTUAL_TERMINAL_PROCESSING
        kernel32.SetConsoleMode(handle, new_mode)
    except Exception:
        return


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Watch a trained Snake agent play in real time (terminal/ANSI).")
    p.add_argument("--model", type=Path, default=None, help="Path to model .zip or .pkl (optional).")
    p.add_argument("--model-dir", type=Path, default=Path("."), help="Directory to search for model files.")

    p.add_argument("--width", type=int, default=10)
    p.add_argument("--height", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episodes", type=int, default=1)

    p.add_argument("--fps", type=float, default=12.0)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--no-ansi", action="store_true", help="Disable ANSI cursor control/clearing.")
    return p.parse_args()


class NeatAgent:
    def __init__(self, genome_path: Path):
        import pickle
        import neat
        from snake7.train_neat import select_action

        self.genome_path = genome_path
        with open(genome_path, "rb") as f:
            self.genome = pickle.load(f)

        # Look for config next to genome
        config_path = genome_path.parent / "neat_config.txt"
        if not config_path.exists():
            config_path = genome_path.parent / "snake_neat_config.txt"

        if not config_path.exists():
            raise FileNotFoundError(f"NEAT config not found next to genome at {genome_path}")

        self.config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            str(config_path),
        )
        self.net = neat.nn.FeedForwardNetwork.create(self.genome, self.config)
        self.select_action = select_action

    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        import numpy as np
        outputs = self.net.activate(obs.tolist())
        action = self.select_action(outputs)
        return np.array(action), None


def main() -> None:
    args = _parse_args()

    from snake7.env import SnakeEnv

    if args.episodes <= 0:
        raise SystemExit("--episodes must be > 0")

    model_path = resolve_model_path(model_path=args.model, model_dir=args.model_dir)

    if model_path.suffix == ".pkl":
        try:
            import neat
        except ImportError:
            raise SystemExit("neat-python is required for .pkl models: pip install neat-python")
        model = NeatAgent(model_path)
    else:
        try:
            from sb3_contrib import RecurrentPPO
        except ImportError:
            raise SystemExit("sb3-contrib is required for .zip models: pip install sb3-contrib")
        model = RecurrentPPO.load(str(model_path))

    use_ansi = (not args.no_ansi) and sys.stdout.isatty()
    if use_ansi:
        _enable_windows_vt_mode()

    delay = 0.0 if args.fps <= 0 else 1.0 / args.fps

    cursor_hidden = False
    try:
        if use_ansi:
            sys.stdout.write("\x1b[?25l")
            sys.stdout.flush()
            cursor_hidden = True

        for ep in range(args.episodes):
            env = SnakeEnv(
                width=args.width,
                height=args.height,
                max_steps=args.max_steps,
                render_mode="ansi",
            )
            obs, info = env.reset(seed=args.seed + ep)
            lstm_state = None
            episode_start = [True]

            frame = env.render()
            frame_text = f"model={model_path.name} episode={ep + 1}/{args.episodes}\n{frame}"
            if use_ansi:
                sys.stdout.write("\x1b[2J\x1b[H")
                sys.stdout.write(frame_text)
                sys.stdout.flush()
            else:
                print(frame_text)

            while True:
                action, lstm_state = model.predict(
                    obs,
                    state=lstm_state,
                    episode_start=episode_start,
                    deterministic=args.deterministic,
                )
                obs, reward, terminated, truncated, info = env.step(int(action))
                episode_start = [terminated or truncated]

                frame = env.render()
                frame_text = f"model={model_path.name} episode={ep + 1}/{args.episodes}\n{frame}"

                if use_ansi:
                    sys.stdout.write("\x1b[H\x1b[J")
                    sys.stdout.write(frame_text)
                    sys.stdout.flush()
                else:
                    print(frame_text)

                if delay:
                    time.sleep(delay)

                if terminated or truncated:
                    if not use_ansi:
                        print(f"done: score={info.get('score')} steps={info.get('steps')}")
                    break
    finally:
        if cursor_hidden:
            sys.stdout.write("\x1b[?25h")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
