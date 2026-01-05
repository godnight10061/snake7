from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class WallTimeEarlyStopCallback(BaseCallback):
    """
    Stop training if a rolling mean episode reward does not improve for a given
    wall-clock duration.

    Improvement metric: mean of the last `window_size` finished-episode rewards
    observed via Monitor/VecMonitor info["episode"]["r"].
    """

    def __init__(
        self,
        *,
        patience_seconds: float = 300.0,
        window_size: int = 20,
        min_delta: float = 0.0,
        save_best_path: Optional[Path] = None,
        stop_score_threshold: Optional[float] = None,
        time_fn: Callable[[], float] = time.monotonic,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        if patience_seconds < 0:
            raise ValueError("patience_seconds must be >= 0")
        if window_size <= 0:
            raise ValueError("window_size must be > 0")
        if min_delta < 0:
            raise ValueError("min_delta must be >= 0")

        self.patience_seconds = float(patience_seconds)
        self.window_size = int(window_size)
        self.min_delta = float(min_delta)
        self.save_best_path = Path(save_best_path) if save_best_path is not None else None
        self.stop_score_threshold = float(stop_score_threshold) if stop_score_threshold is not None else None
        self.time_fn = time_fn

        self.recent_rewards: deque[float] = deque(maxlen=self.window_size)
        self.best_score: float = float("-inf")
        self.last_improvement_time: Optional[float] = None
        self.stopped: bool = False

    def _on_training_start(self) -> None:
        return None

    def _on_step(self) -> bool:
        if self.stopped:
            return False

        infos = self.locals.get("infos", None)
        if infos is None:
            return True

        now = float(self.time_fn())
        info_list = [infos] if isinstance(infos, dict) else infos

        updated = False
        for info in info_list:
            if not isinstance(info, dict):
                continue
            ep = info.get("episode")
            if not isinstance(ep, dict):
                continue
            r = ep.get("r", None)
            if r is None:
                continue
            try:
                reward = float(r)
            except (TypeError, ValueError):
                continue
            self.recent_rewards.append(reward)
            updated = True

        if updated and self.recent_rewards:
            score = float(np.mean(self.recent_rewards))
            if self.last_improvement_time is None:
                self.last_improvement_time = now
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.last_improvement_time = now
                if self.save_best_path is not None:
                    self.save_best_path.parent.mkdir(parents=True, exist_ok=True)
                    self.model.save(str(self.save_best_path))
                if self.verbose:
                    print(f"[early_stop] new best mean_reward={score:.4f}")
                if self.stop_score_threshold is not None and score > self.stop_score_threshold:
                    self.stopped = True
                    if self.verbose:
                        print(
                            f"[early_stop] stopping: mean_reward={score:.4f} > threshold={self.stop_score_threshold}"
                        )
                    return False

        if self.last_improvement_time is None:
            return True

        if now - self.last_improvement_time > self.patience_seconds:
            self.stopped = True
            if self.verbose:
                dt = now - self.last_improvement_time
                print(f"[early_stop] stopping: no improvement for {dt:.1f}s (patience={self.patience_seconds}s)")
            return False

        return True
