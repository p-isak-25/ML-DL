from argparse import ArgumentParser

import gymnasium as gym
from stable_baselines3 import A2C


def main():
    parser = ArgumentParser(description="Train and evaluate A2C on CartPole.")
    parser.add_argument("--timesteps", type=int, default=10_000)
    parser.add_argument("--eval-steps", type=int, default=1000)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    with gym.make("CartPole-v1") as env:
        model = A2C("MlpPolicy", env, verbose=1, device="cpu")
        model.learn(total_timesteps=args.timesteps)

    with gym.make("CartPole-v1", render_mode=None if args.headless else "human") as env:
        observation, _ = env.reset()
        for _ in range(args.eval_steps):
            action, _ = model.predict(observation, deterministic=True)
            observation, _, terminated, truncated, _ = env.step(action.item())
            if terminated or truncated:
                observation, _ = env.reset()


if __name__ == "__main__":
    main()
