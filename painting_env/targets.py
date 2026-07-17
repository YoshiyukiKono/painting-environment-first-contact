from __future__ import annotations

import numpy as np


def make_target(size: int = 64) -> np.ndarray:
    """Create a simple movie-sign-like target with large readable shapes."""
    canvas = np.ones((size, size, 3), dtype=np.float32)

    yy, xx = np.mgrid[:size, :size]

    # Background
    canvas[:] = np.array([0.92, 0.78, 0.52], dtype=np.float32)

    # Large dark circular head
    cx, cy = size * 0.50, size * 0.43
    head = (xx - cx) ** 2 + (yy - cy) ** 2 <= (size * 0.22) ** 2
    canvas[head] = np.array([0.10, 0.12, 0.16], dtype=np.float32)

    # Face area
    face = ((xx - cx) / (size * 0.15)) ** 2 + ((yy - size * 0.46) / (size * 0.18)) ** 2 <= 1.0
    canvas[face] = np.array([0.88, 0.56, 0.39], dtype=np.float32)

    # Shirt/body
    body = (
        (yy >= size * 0.62)
        & (np.abs(xx - cx) <= (yy - size * 0.55) * 0.65)
    )
    canvas[body] = np.array([0.12, 0.28, 0.65], dtype=np.float32)

    # Two large highlights
    for eye_x in (size * 0.45, size * 0.55):
        eye = (xx - eye_x) ** 2 + (yy - size * 0.44) ** 2 <= (size * 0.025) ** 2
        canvas[eye] = np.array([0.96, 0.96, 0.90], dtype=np.float32)

    return np.clip(canvas, 0.0, 1.0)
