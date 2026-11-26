import gymnasium as gym
import numpy as np

# 새롭게 Policy Iteration 클래스 정의
class PolicyIteration:
    def __init__(self, env):
        self.env = env 
        self.state_size = env.observation_space.n # 상태의 총 개수 = 500 
        self.action_size = env.action_space.n # 행동의 총 개수 = 6 
 
        # 초기 정책: 모든 행동에 대해 균등한 확률 (랜덤 초기화)
        self.policy_table = np.ones([self.state_size, self.action_size]) / self.action_size 
        # 초기 가치 함수는 모든 상태에 대해 0으로 설정
        self.value_table = np.zeros(self.state_size)

    # 수정된 Policy Evaluation (theta와 delta 반영)
    def policy_evaluation(self, discount_factor=0.9, theta=0.00001):
        while True:
            delta = 0  # 가치 변화의 최대값
            new_value_table = np.zeros(self.state_size) # 새로운 value 저장용 테이블
            
            for state in range(self.state_size):
                value = 0 #해당 상태의 새로운 가치 계산을 위한 변수
                for action in range(self.action_size):
                    #현재 정책에서의 행동 선택 확률률
                    action_prob = self.policy_table[state][action]
                    for prob, next_state, reward, done in self.env.unwrapped.P[state][action]:
                         #벨만 기대방정식
                         #현재 상태의 가치 = 기대 보상 + 다음 상태의 할인된 가치
                         value += action_prob * prob * (reward + discount_factor * self.value_table[next_state])
                #새 가치 업데이트트
                new_value_table[state] = value
                delta = max(delta, abs(self.value_table[state] - value)) # 변화량 중 최대값 업업데이트
            self.value_table = new_value_table.copy() #가치 함수 업데이트트
            if delta < theta:  # 변화량이 theta 이하이면 수렴으로 여기고 종료료
                break

    # Policy Improvement (policy_stable 체크 포함)
    def policy_improvement(self):
        policy_stable = True # 정책이 바뀌는지 확인 flag
        for state in range(self.state_size):
            # 기존 정책에서 선택된 action
            old_action = np.argmax(self.policy_table[state])
            # 행동 별 기대 가치 저장
            action_values = np.zeros(self.action_size)

            for action in range(self.action_size):
                for prob, next_state, reward, done in self.env.unwrapped.P[state][action]:
                    #현재 상태에서의 행동 가치를 계산
                    action_values[action] += prob * (reward + self.value_table[next_state])
            #가장 높은 가치를 가는 행동 선택택
            best_action = np.argmax(action_values)
            self.policy_table[state] = np.eye(self.action_size)[best_action] # 정책 얻데이트
            if old_action != best_action:
                policy_stable = False #정책이 바뀌었다 -> 수렴 x
        return policy_stable # 정책 수렴 여부 반환
    
    #특정 상태의 가치 반환
    def get_policy(self, state):
        return self.policy_table[state]

    #특정 상태의 가치 반환
    def get_value(self, state):
        return self.value_table[state]

# 알고리즘 실행 함수
def run_policy_iteration(env, discount_factor=0.9, theta=0.00001, max_iterations=1000):
    agent = PolicyIteration(env)

    for i in range(max_iterations):
        # 현재 정책을 바탕으로 가치 함수 평가: delta < theta 만족할때까지 반복
        agent.policy_evaluation(discount_factor=discount_factor, theta=theta)

        # 가치 함수 기반으로 정책 개선
        policy_stable = agent.policy_improvement()
        print(f"[Iteration {i+1}] 정책 안정화 여부: {policy_stable}")
        
        # 정책이 더 이상 바뀌지 않으면 종료
        if policy_stable:
            print(f"[종료] 정책이 안정화되었습니다. {i+1}번째 iteration에서 수렴.")
            break
    return agent

# 실행
env = gym.make("Taxi-v3", render_mode=None)
trained_agent = run_policy_iteration(env)
