from pathlib import Path

from PIL import Image

from painting_env import PaintingEnv
from painting_env.agents import greedy_action


def main() -> None:
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    env = PaintingEnv()
    _, info = env.reset(seed=7)

    frames: list[Image.Image] = []
    total_reward = 0.0

    for step in range(env.max_steps):
        action = greedy_action(env, candidates=192)
        _, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if step % 5 == 0:
            frames.append(Image.fromarray(env.render()))

        if terminated or truncated:
            break

    final_image = Image.fromarray(env.render())
    final_image.save(output_dir / "greedy_result.png")

    if frames:
        frames[0].save(
            output_dir / "greedy_progress.gif",
            save_all=True,
            append_images=frames[1:],
            duration=90,
            loop=0,
        )

    print(f"steps={info['step']}")
    print(f"final_error={info['error']:.6f}")
    print(f"total_reward={total_reward:.6f}")
    print("saved: outputs/greedy_result.png")
    print("saved: outputs/greedy_progress.gif")


if __name__ == "__main__":
    main()
