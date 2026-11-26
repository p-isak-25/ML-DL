
import gymnasium as gym

# Taxi-v3 환경 생성
env = gym.make("Taxi-v3", render_mode="human")

# 총 20 에피소드 실행
for n_episode in range(20):
    observation, info = env.reset()  # 환경 초기화
    for t in range(100):
        env.render()  # 환경 시각화
        print("현재 상태:", observation)  # 상태 출력
        action = env.action_space.sample()  # 무작위 액션 선택
        observation, reward, terminated, truncated, info = env.step(action)  # 액션 실행
        if terminated or truncated:  # 에피소드가 종료되면 루프 탈출출
            print("Episode finished after {} timesteps".format(t + 1))
            break

env.close()
