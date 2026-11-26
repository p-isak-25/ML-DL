import numpy as np
import gymnasium as gym
if __package__:
    from .policyiteration import PolicyIteration
else:
    from policyiteration import PolicyIteration


# Taxi-v3 환경 생성 (500개 상태, 6개 행동)
env = gym.make("Taxi-v3")
max_iterations = 100 # 최대 반복 횟수 임의로 설정정

# 방법 1: Eval 10번 + Improve 1번을 반복
agent1 = PolicyIteration(env)
for i in range(max_iterations):
    # 현재 정책에 대한 평가 10번 반복복
    agent1.policy_evaluation(iter_num=10, discount_factor=0.9)
    # 정책 improve
    if agent1.policy_improvement():
        # 정책이 더 이상 바뀌지 않으면 수렴 -> 반복 종료 
        print(f"[Agent1] 정책 수렴: {i+1}번째 반복에서 종료")
        break

# 전략 2: Eval 1번 + Improve 1번
agent2 = PolicyIteration(env)
for i in range(max_iterations):
    # 정책 평가를 1번만 
    agent2.policy_evaluation(iter_num=1, discount_factor=0.9)
    # 정책 improve
    if agent2.policy_improvement():
        print(f"[Agent2] 정책 수렴: {i+1}번째 반복에서 종료")
        break


# 전체 상태에 대한 결과 비교
same_value_count = 0 # 가치 함수가 거의 같은 상태 수
same_policy_count = 0 # 정책이 같은 상태 수

print()
print("[전체 상태에 대한 정책 및 가치 비교]")
print("State | Value1    Value2    | Policy1  Policy2  | SamePolicy")
print("-------------------------------------------------------------")

for state in range(env.observation_space.n):  # 모든 상태(0~499) 비교
    v1 = agent1.get_value(state)  # 방법 1 의 가치 함수 값
    v2 = agent2.get_value(state)  # 방법 2 의 가치 함수 값
    p1 = np.argmax(agent1.get_policy(state)) # 방법 1 의 best action
    p2 = np.argmax(agent2.get_policy(state)) # 방법법 2 의 best action
    same_policy = p1 == p2 # 두 방법법의 정책이 같은지 확인인

    # 부동소수점 오차 고려하여 value 유사성 확인 -> np.isclose 이용
    if np.isclose(v1, v2): 
        same_value_count += 1
    if same_policy:
        same_policy_count += 1

    # 상태번호, 두 방법법의 value, 정책, 정책 일치 여부
    print(f"{state:5} | {v1:9.2f} {v2:9.2f} | {p1:^8} {p2:^8} | {same_policy}")


# 요약
print("\n[정책 일치율 및 가치 일치율 요약]")
print(f"가치 함수가 거의 같은 상태 수 : {same_value_count} / 500")
print(f"정책이 같은 상태 수 : {same_policy_count} / 500")
