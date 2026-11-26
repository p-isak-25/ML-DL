import gymnasium as gym
if __package__:
    from .policyiteration import PolicyIteration
else:
    from policyiteration import PolicyIteration
import numpy as np

# Taxi-v3 환경 생성
env = gym.make("Taxi-v3", render_mode= None)

# PolicyIteration 객체 생성
agent = PolicyIteration(env)


# 정책 반복 실행 함수
def run_policy_iteration(agent, max_iterations=1000, discount_factor = 0.9):
    for i in range(max_iterations):
        # 현재 정책 바탕, 상태 가치 함수를 평가(벨만 기대 방정식 기반반)
        agent.policy_evaluation(iter_num=100, discount_factor=discount_factor)  
        # 평가한 가치 함수를 기반으로 정책 개선
        policy_stable = agent.policy_improvement() 
        # 현재 반복 정보 츌력하기
        print(f"[Iteration {i+1}] 정책 안정화 여부: {policy_stable}")
        
        # 정책이 더 이상 바뀌지 않으면 반복을 종료한다
        if policy_stable:
            print("정책이 안정화되었습니다. 알고리즘 종료.")
            break
    return agent


# 알고리즘 실행
trained_agent = run_policy_iteration(agent, discount_factor=0.9)


# 결과 출력
print()
print("[전체 상태에 대한 정책 및 가치]")
for state in range(trained_agent.state_size): #state_size = 500 (taxi - v3)
    policy = trained_agent.get_policy(state) # 해당 상태에서의 정책
    value = trained_agent.get_value(state)  # 해당 상태의 가치 함수 값
    print(f"State {state} - Policy: {policy}, Value: {value:.2f}")
