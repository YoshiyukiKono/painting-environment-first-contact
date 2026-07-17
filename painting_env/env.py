from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .targets import make_target


DEFAULT_PALETTE = np.array(
    [
        [0.96, 0.96, 0.90],  # warm white
        [0.10, 0.12, 0.16],  # near black
        [0.88, 0.56, 0.39],  # skin/orange
        [0.12, 0.28, 0.65],  # blue
        [0.92, 0.78, 0.52],  # ochre
    ],
    dtype=np.float32,
)


@dataclass(frozen=True)
class PaintAction:
    x: int
    y: int
    color_index: int
    radius: int


class PaintingEnv(gym.Env):
    """Minimal painting environment using opaque circular stamps."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        size: int = 64,
        max_steps: int = 250,
        min_radius: int = 2,
        max_radius: int = 8,
        target: np.ndarray | None = None,
        palette: np.ndarray | None = None,
    ) -> None:
        super().__init__()

        if size < 8:
            raise ValueError("size must be at least 8")
        if min_radius < 1 or max_radius < min_radius:
            raise ValueError("invalid radius range")

        self.size = size
        self.max_steps = max_steps
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.palette = np.asarray(
            DEFAULT_PALETTE if palette is None else palette,
            dtype=np.float32,
        )

        self.target = (
            make_target(size)
            if target is None
            else np.asarray(target, dtype=np.float32)
        )
        if self.target.shape != (size, size, 3):
            raise ValueError(f"target must have shape {(size, size, 3)}")

        # MultiDiscrete values:
        # x, y, color index, radius offset
        self.action_space = spaces.MultiDiscrete(
            [size, size, len(self.palette), max_radius - min_radius + 1]
        )

        self.observation_space = spaces.Dict(
            {
                "canvas": spaces.Box(
                    low=0.0, high=1.0, shape=(size, size, 3), dtype=np.float32
                ),
                "target": spaces.Box(
                    low=0.0, high=1.0, shape=(size, size, 3), dtype=np.float32
                ),
            }
        )

        self.canvas = np.ones((size, size, 3), dtype=np.float32)
        self.steps = 0

    def _error(self) -> float:
        return float(np.mean((self.canvas - self.target) ** 2))

    def _observation(self) -> dict[str, np.ndarray]:
        return {
            "canvas": self.canvas.copy(),
            "target": self.target.copy(),
        }

    def _decode_action(self, action: np.ndarray | list[int]) -> PaintAction:
        values = np.asarray(action, dtype=np.int64)
        if values.shape != (4,):
            raise ValueError("action must contain x, y, color_index, radius_offset")

        x, y, color_index, radius_offset = values.tolist()
        radius = self.min_radius + radius_offset
        return PaintAction(x=x, y=y, color_index=color_index, radius=radius)

    def apply_action(self, action: np.ndarray | list[int]) -> None:
        paint = self._decode_action(action)
        yy, xx = np.ogrid[: self.size, : self.size]
        mask = (xx - paint.x) ** 2 + (yy - paint.y) ** 2 <= paint.radius**2
        self.canvas[mask] = self.palette[paint.color_index]

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict[str, np.ndarray], dict]:
        super().reset(seed=seed)
        self.canvas = np.ones((self.size, self.size, 3), dtype=np.float32)
        self.steps = 0
        return self._observation(), {"error": self._error()}

    def step(
        self,
        action: np.ndarray | list[int],
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict]:
        before = self._error()
        self.apply_action(action)
        self.steps += 1
        after = self._error()

        reward = before - after
        terminated = after < 0.005
        truncated = self.steps >= self.max_steps

        info = {
            "error": after,
            "improvement": reward,
            "step": self.steps,
        }
        return self._observation(), reward, terminated, truncated, info

    def render(self) -> np.ndarray:
        return (np.clip(self.canvas, 0.0, 1.0) * 255).astype(np.uint8)
