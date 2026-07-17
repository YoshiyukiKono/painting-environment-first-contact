import numpy as np

from painting_env import PaintingEnv


def test_reset_shapes() -> None:
    env = PaintingEnv(size=32)
    observation, info = env.reset(seed=1)

    assert observation["canvas"].shape == (32, 32, 3)
    assert observation["target"].shape == (32, 32, 3)
    assert info["error"] >= 0.0


def test_step_changes_canvas() -> None:
    env = PaintingEnv(size=32)
    observation, _ = env.reset(seed=1)
    before = observation["canvas"].copy()

    observation, reward, terminated, truncated, info = env.step([16, 16, 1, 2])

    assert not np.array_equal(before, observation["canvas"])
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert info["step"] == 1


def test_matching_color_can_improve_reward() -> None:
    target = np.ones((16, 16, 3), dtype=np.float32)
    target[5:12, 5:12] = np.array([0.10, 0.12, 0.16], dtype=np.float32)

    env = PaintingEnv(
        size=16,
        target=target,
        min_radius=2,
        max_radius=4,
    )
    env.reset(seed=1)

    _, reward, _, _, _ = env.step([8, 8, 1, 2])

    assert reward > 0.0
