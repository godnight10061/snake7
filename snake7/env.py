from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces


_DIR_TO_DELTA: dict[int, tuple[int, int]] = {
    0: (0, -1),  # up
    1: (1, 0),  # right
    2: (0, 1),  # down
    3: (-1, 0),  # left
}


class SnakeEnv(gym.Env[np.ndarray, int]):
    metadata = {"render_modes": ["ansi", "human"], "render_fps": 10}

    def __init__(
        self,
        *,
        width: int = 10,
        height: int = 10,
        max_steps: Optional[int] = None,
        max_steps_without_food: Optional[int] = None,
        step_penalty: float = -0.01,
        truncation_penalty: float = 0.0,
        distance_shaping: float = 0.0,
        distance_shaping_clip: float = 0.0,
        food_reward: float = 1.0,
        death_penalty: float = -1.0,
        win_reward: float = 5.0,
        render_mode: Optional[str] = None,
    ) -> None:
        if width < 3 or height < 3:
            raise ValueError("width and height must be >= 3")

        self.width = int(width)
        self.height = int(height)
        self.max_steps = int(max_steps) if max_steps is not None else self.width * self.height * 4
        if self.max_steps <= 0:
            raise ValueError("max_steps must be > 0")

        self.max_steps_without_food = None if max_steps_without_food is None else int(max_steps_without_food)
        if self.max_steps_without_food is not None and self.max_steps_without_food <= 0:
            raise ValueError("max_steps_without_food must be > 0")

        self.step_penalty = float(step_penalty)
        self.truncation_penalty = float(truncation_penalty)
        self.distance_shaping = float(distance_shaping)
        self.distance_shaping_clip = float(distance_shaping_clip)
        self.food_reward = float(food_reward)
        self.death_penalty = float(death_penalty)
        self.win_reward = float(win_reward)
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(9,),
            dtype=np.float32,
        )

        self._snake: list[tuple[int, int]] = []
        self._direction: int = 1
        self._food: Optional[tuple[int, int]] = None
        self._steps: int = 0
        self._steps_since_food: int = 0
        self._score: int = 0

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)

        self._steps = 0
        self._steps_since_food = 0
        self._score = 0

        head_x = max(2, self.width // 2)
        head_y = self.height // 2
        self._direction = 1  # right
        self._snake = [(head_x, head_y), (head_x - 1, head_y), (head_x - 2, head_y)]
        self._place_food()

        obs = self._get_obs()
        info = self._get_info()
        if self.render_mode == "human":
            print(self._render_ansi())
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        self._steps += 1

        if action == 1:
            self._direction = (self._direction - 1) % 4
        elif action == 2:
            self._direction = (self._direction + 1) % 4

        head_x, head_y = self._snake[0]
        prev_food = self._food
        prev_dist = 0
        if prev_food is not None:
            fx, fy = prev_food
            prev_dist = abs(fx - head_x) + abs(fy - head_y)

        dx, dy = _DIR_TO_DELTA[self._direction]
        new_head = (head_x + dx, head_y + dy)

        will_eat = prev_food is not None and new_head == prev_food
        body_to_check = self._snake if will_eat else self._snake[:-1]
        collision = self._is_wall(new_head) or new_head in body_to_check

        reward = self.step_penalty
        terminated = False
        truncated = False

        if collision:
            reward = self.death_penalty
            terminated = True
        else:
            self._snake.insert(0, new_head)
            if will_eat:
                self._score += 1
                self._steps_since_food = 0
                reward += self.food_reward
                if len(self._snake) == self.width * self.height:
                    terminated = True
                    reward += self.win_reward
                    self._food = None
                else:
                    self._place_food()
            else:
                self._steps_since_food += 1
                self._snake.pop()

            if self.distance_shaping != 0.0 and prev_food is not None:
                fx, fy = prev_food
                new_dist = abs(fx - new_head[0]) + abs(fy - new_head[1])
                shaping = self.distance_shaping * float(prev_dist - new_dist)
                if self.distance_shaping_clip > 0:
                    clip = float(self.distance_shaping_clip)
                    shaping = max(-clip, min(clip, shaping))
                reward += shaping

            if not terminated and self._steps >= self.max_steps:
                truncated = True

            if not terminated and not truncated and self.max_steps_without_food is not None:
                if self._steps_since_food >= self.max_steps_without_food:
                    truncated = True

            if truncated:
                reward += self.truncation_penalty

        obs = self._get_obs()
        info = self._get_info()
        if self.render_mode == "human":
            print(self._render_ansi())
        return obs, float(reward), bool(terminated), bool(truncated), info

    def render(self) -> Optional[str]:
        if self.render_mode is None:
            return None
        if self.render_mode == "ansi":
            return self._render_ansi()
        if self.render_mode == "human":
            print(self._render_ansi())
            return None
        raise ValueError(f"Unsupported render_mode: {self.render_mode}")

    def close(self) -> None:
        return None

    def _get_info(self) -> dict[str, Any]:
        return {
            "score": self._score,
            "length": len(self._snake),
            "steps": self._steps,
            "steps_since_food": self._steps_since_food,
        }

    def _get_obs(self) -> np.ndarray:
        head_x, head_y = self._snake[0]
        if self._food is None:
            dx = 0.0
            dy = 0.0
        else:
            fx, fy = self._food
            dx = (fx - head_x) / (self.width - 1)
            dy = (fy - head_y) / (self.height - 1)

        obs = np.zeros((9,), dtype=np.float32)
        obs[0] = np.float32(dx)
        obs[1] = np.float32(dy)
        obs[2 + self._direction] = 1.0
        obs[6] = 1.0 if self._would_collide(self._direction) else 0.0
        obs[7] = 1.0 if self._would_collide((self._direction - 1) % 4) else 0.0
        obs[8] = 1.0 if self._would_collide((self._direction + 1) % 4) else 0.0
        return obs

    def _is_wall(self, pos: tuple[int, int]) -> bool:
        x, y = pos
        return x < 0 or x >= self.width or y < 0 or y >= self.height

    def _would_collide(self, direction: int) -> bool:
        head_x, head_y = self._snake[0]
        dx, dy = _DIR_TO_DELTA[direction]
        new_head = (head_x + dx, head_y + dy)

        will_eat = self._food is not None and new_head == self._food
        body_to_check = self._snake if will_eat else self._snake[:-1]
        return self._is_wall(new_head) or new_head in body_to_check

    def _place_food(self) -> None:
        occupied = set(self._snake)
        empties: list[tuple[int, int]] = [
            (x, y)
            for y in range(self.height)
            for x in range(self.width)
            if (x, y) not in occupied
        ]
        if not empties:
            self._food = None
            return
        idx = int(self.np_random.integers(0, len(empties)))
        self._food = empties[idx]

    def _render_ansi(self) -> str:
        grid = [["." for _ in range(self.width)] for _ in range(self.height)]
        if self._food is not None:
            fx, fy = self._food
            grid[fy][fx] = "*"
        for i, (x, y) in enumerate(self._snake):
            grid[y][x] = "H" if i == 0 else "o"
        rows = ["".join(r) for r in grid]
        header = f"score={self._score} length={len(self._snake)} steps={self._steps}/{self.max_steps}"
        return header + "\n" + "\n".join(rows) + "\n"
