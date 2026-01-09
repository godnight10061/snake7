from __future__ import annotations

import argparse
import os
import sys
import time
import zipfile
from pathlib import Path
from typing import Optional, Union

# Allow running this file directly (e.g. `python snake7/watch.py`) by ensuring the
# repository root is on sys.path. Recommended usage is still `python -m snake7.watch`.
if __package__ in (None, "") and __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    __package__ = "snake7"


def is_recurrent_model(model_path: Path) -> bool:
    """
    Check if the model at model_path was saved with RecurrentPPO.
    Peeks into the ZIP's 'data' file without fully loading the model.
    """
    try:
        with zipfile.ZipFile(model_path, "r") as archive:
            if "data" not in archive.namelist():
                return False
            data = archive.read("data")
            # RecurrentPPO models in sb3_contrib contain these strings in their pickled/json data.
            return b"RecurrentPPO" in data
    except Exception:
        return False


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

        self.genome_path = Path(genome_path)
        with open(self.genome_path, "rb") as f:
            self.genome = pickle.load(f)

        config_candidates = [
            self.genome_path.parent / "neat_config.txt",
            self.genome_path.parent / "snake_neat_config.txt",
        ]
        config_path = next((p for p in config_candidates if p.exists()), None)
        if config_path is None:
            expected = ", ".join(p.name for p in config_candidates)
            raise FileNotFoundError(
                f"NEAT config not found next to genome: {self.genome_path} (expected {expected})"
            )

        self.config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            str(config_path),
        )

        if bool(getattr(self.config.genome_config, "feed_forward", True)):
            self.net = neat.nn.FeedForwardNetwork.create(self.genome, self.config)
        else:
            self.net = neat.nn.RecurrentNetwork.create(self.genome, self.config)

    @staticmethod
    def _select_action(outputs) -> int:
        return max(range(len(outputs)), key=lambda i: float(outputs[i]))

    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        import numpy as np

        if episode_start and bool(episode_start[0]) and hasattr(self.net, "reset"):
            try:
                self.net.reset()
            except Exception:
                pass

        outputs = self.net.activate(obs.tolist())
        action = self._select_action(outputs)
        return np.array(action), None


def main() -> None:
    args = _parse_args()

    if args.episodes <= 0:
        raise SystemExit("--episodes must be > 0")

    model_path = resolve_model_path(model_path=args.model, model_dir=args.model_dir)

    if model_path.suffix.lower() in {".pkl", ".pickle"}:
        try:
            import neat  # noqa: F401
        except ImportError:
            raise SystemExit(
                "neat-python is required for .pkl models: python -m pip install neat-python\n"
                "Or install all dev dependencies: pip install -e .[dev]"
            )
        model = NeatAgent(model_path)
    else:
        try:
            from stable_baselines3 import PPO
        except ImportError:
            raise SystemExit(
                "stable-baselines3 is required for .zip models: python -m pip install stable-baselines3\n"
                "Or install all dev dependencies: pip install -e .[dev]"
            )

        try:
            from sb3_contrib import RecurrentPPO
        except ImportError:
            RecurrentPPO = None

        # Robust model loading: detect if RecurrentPPO is needed.
        if is_recurrent_model(model_path):
            if RecurrentPPO is None:
                raise SystemExit("Model requires sb3_contrib.RecurrentPPO but it is not installed.")
            try:
                model = RecurrentPPO.load(str(model_path))
            except Exception as e:
                raise SystemExit(f"Could not load model as RecurrentPPO: {e}")
        else:
            try:
                model = PPO.load(str(model_path))
            except Exception as e:
                # Fallback to RecurrentPPO if PPO fails and it's installed, just in case detection missed it.
                if RecurrentPPO is not None:
                    try:
                        model = RecurrentPPO.load(str(model_path))
                    except Exception:
                        raise SystemExit(f"Could not load model as PPO or RecurrentPPO: {e}")
                else:
                    raise SystemExit(f"Could not load model as PPO: {e}")

    use_ansi = (not args.no_ansi) and sys.stdout.isatty()
    if use_ansi:
        _enable_windows_vt_mode()

    delay = 0.0 if args.fps <= 0 else 1.0 / args.fps

    from snake7.env import SnakeEnv
    from snake7.wrappers import ObsStackWrapper

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

            # Check if we need to wrap the environment (e.g. for Transformer models)
            model_obs = getattr(model, "observation_space", None)
            model_shape = getattr(model_obs, "shape", None)
            if model_shape is not None and len(model_shape) == 2 and len(env.observation_space.shape) == 1:
                n_stack = model_shape[0]
                env = ObsStackWrapper(env, n_stack=n_stack)

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
