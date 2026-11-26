from argparse import ArgumentParser

import ale_py
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper


def make_env(render_mode=None):
    return AtariWrapper(
        gym.make("ALE/Breakout-v5", frameskip=1, render_mode=render_mode),
        terminal_on_life_loss=False,
    )


def main():
    parser = ArgumentParser(description="Train and evaluate DQN on Atari Breakout.")
    parser.add_argument("--timesteps", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--eval-steps", type=int, default=1000)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    gym.register_envs(ale_py)
    with make_env() as env:
        model = DQN(
            "CnnPolicy",
            env,
            learning_rate=5e-4,
            buffer_size=15_000,
            learning_starts=200,
            batch_size=32,
            gamma=0.8,
            train_freq=1,
            gradient_steps=1,
            target_update_interval=50,
            verbose=1,
            tensorboard_log="logs/dqn_cnn/",
            device="cpu",
        )
        model.learn(total_timesteps=args.timesteps)
        model.save("dqn_atari")

    with make_env(render_mode=None if args.headless else "human") as env:
        for episode in range(args.eval_episodes):
            observation, _ = env.reset()
            for _ in range(args.eval_steps):
                action, _ = model.predict(observation, deterministic=True)
                observation, _, terminated, truncated, _ = env.step(action.item())
                if terminated or truncated:
                    break
            print(f"Episode {episode + 1} finished.")


if __name__ == "__main__":
    main()
