from __future__ import annotations

import numpy as np

from .env import PaintingEnv


def random_action(env: PaintingEnv) -> np.ndarray:
    return env.action_space.sample()


def greedy_action(
    env: PaintingEnv,
    candidates: int = 128,
) -> np.ndarray:
    """Select the best action among randomly sampled candidates.

    This is deliberately simple. It proves that the Environment has a usable
    reward signal before introducing a learned policy.
    """
    original = env.canvas.copy()
    baseline_error = env._error()

    best_action = None
    best_error = baseline_error

    for _ in range(candidates):
        action = env.action_space.sample()
        env.canvas = original.copy()
        env.apply_action(action)
        error = env._error()

        if error < best_error:
            best_error = error
            best_action = action.copy()

    env.canvas = original

    if best_action is None:
        return random_action(env)
    return best_action
