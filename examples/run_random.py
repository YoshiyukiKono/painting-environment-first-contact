from pathlib import Path

from PIL import Image

from painting_env import PaintingEnv
from painting_env.agents import random_action


def main() -> None:
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    env = PaintingEnv()
    _, info = env.reset(seed=7)

    total_reward = 0.0
    for _ in range(env.max_steps):
        action = random_action(env)
        _, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break

    Image.fromarray(env.render()).save(output_dir / "random_result.png")
    Image.fromarray((env.target * 255).astype("uint8")).save(
        output_dir / "target.png"
    )

    print(f"steps={info['step']}")
    print(f"final_error={info['error']:.6f}")
    print(f"total_reward={total_reward:.6f}")
    print("saved: outputs/random_result.png")


if __name__ == "__main__":
    main()
