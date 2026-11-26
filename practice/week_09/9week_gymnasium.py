import gymnasium as gym

env = gym.make('CartPole-v1', render_mode = "human")

for n_episodes in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action) 
        #<실습> 종료 시 다음 에피소드 진행하는 코드
        if terminated or truncated:
            print(f"Episode {n_episodes + 1} ended after {t + 1} timesteps\n")
            break
env.close()