from __future__ import annotations

from collections import deque
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class ObsStackWrapper(gym.Wrapper):
    """
    Stack the last N vector observations into a (N, obs_dim) tensor.

    Useful for non-recurrent policies that still want short-term memory (e.g. a Transformer encoder).
    """

    def __init__(self, env: gym.Env, *, n_stack: int = 8) -> None:
        super().__init__(env)
        if n_stack <= 0:
            raise ValueError("n_stack must be > 0")

        obs_space = env.observation_space
        if not isinstance(obs_space, spaces.Box) or obs_space.shape is None or len(obs_space.shape) != 1:
            raise ValueError("ObsStackWrapper expects a 1D Box observation space")

        self.n_stack = int(n_stack)
        self._obs_dtype = obs_space.dtype

        low = np.repeat(obs_space.low[None, :], self.n_stack, axis=0)
        high = np.repeat(obs_space.high[None, :], self.n_stack, axis=0)
        self.observation_space = spaces.Box(low=low, high=high, dtype=self._obs_dtype)

        self._frames: deque[np.ndarray] = deque(maxlen=self.n_stack)

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        obs_arr = np.asarray(obs, dtype=self._obs_dtype)

        self._frames.clear()
        for _ in range(self.n_stack):
            self._frames.append(obs_arr.copy())

        return self._get_obs(), info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs_arr = np.asarray(obs, dtype=self._obs_dtype)

        self._frames.append(obs_arr.copy())

        return self._get_obs(), float(reward), bool(terminated), bool(truncated), info

    def _get_obs(self) -> np.ndarray:
        if len(self._frames) != self.n_stack:
            raise RuntimeError("ObsStackWrapper internal buffer is not initialized; call reset() first")
        return np.stack(list(self._frames), axis=0)
