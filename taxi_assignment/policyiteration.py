import numpy as np
import gymnasium as gym  # OpenAI Gymnasium 패키지 (Taxi-v3 환경 포함)

# PolicyIteration 클래스 정의
class PolicyIteration:
    # PolicyIteration class의 생성자
    def __init__(self, env):
        self.env = env
        self.state_size = env.observation_space.n     # 상태 개수(상태 공간 크기)->(Taxi-v3는 500)
        self.action_size = env.action_space.n         # 행동 개수(행동 공간 크기)-> (Taxi-v3는 6)
        # 초기 정책은 모든 행동을 균등 확률로 선택
        self.policy_table = np.ones([self.state_size, self.action_size]) / self.action_size
        # 모든 상태에 대한 초기 가치 함수는 0
        self.value_table = np.zeros(self.state_size)

    # Policy Evaluation 함수
    #-> 현재 정책에 따라 상태 가치 함수 계산
    def policy_evaluation(self, iter_num, discount_factor):
        for _ in range(iter_num):  # 지정된 횟수만큼 반복
            new_value_table = np.zeros(self.state_size) # 새로운 가치 테이블 초기화
            for state in range(self.state_size): # 모든 상태를 
                value = 0 # 상태 초기화화
                for action in range(self.action_size): # 가능한 모든 행동
                    action_prob = self.policy_table[state][action]  # 현재 정책에서의 행동 선택 확률
                    # 환경의 P 테이블에서 다음 상태, 보상, 전이 확률 정보를 가져온것
                    for prob, next_state, reward, done in self.env.unwrapped.P[state][action]:
                        # 벨만 기대 방정식 적용: 보상 + 감가된 다음 상태의 가치
                        value += action_prob * prob * (reward + discount_factor * self.value_table[next_state])
                new_value_table[state] = value  # 업데이트된 상태 가치 저장
            self.value_table = new_value_table.copy() # 가치 업데이트

    # Policy Improvement 함수
    #-> 가치 함수 기준으로 더 나은 정책 생성성
    def policy_improvement(self):
        policy_stable = True # 정책이 바뀌었는지 check하는 용도도
        for state in range(self.state_size):
            old_action = np.argmax(self.policy_table[state])  # 기존 최선 행동
            action_values = np.zeros(self.action_size) # 행동별 Q값(각 상태에서 가능한 행동마다 해당 행동의 기대가치) 저장
            for action in range(self.action_size):
                # 가치 함수 기반으로 행동 가치 계산산    
                for prob, next_state, reward, done in self.env.unwrapped.P[state][action]:
                    action_values[action] += prob * (reward + self.value_table[next_state])
            best_action = np.argmax(action_values) # 가장 가치가 높은 행동 선택
            # 정책 테이블을 단일 best action으로 업데이트 (greedy policy)
            self.policy_table[state] = np.eye(self.action_size)[best_action]
            # 정책이 바뀌었는지 확인
            if old_action != best_action:
                policy_stable = False
        return policy_stable  # 바뀐 것이 없다면 True 반환

    # 특정 상태의 정책 반환
    def get_policy(self, state):
        return self.policy_table[state] 

    # 특정 상태의 가치 반환
    def get_value(self, state):
        return self.value_table[state] 

    
    # 현재 value_table 출력
    def print_value(self):
        print("[ Value Table ]")
        for state in range(self.state_size):
            print(f"State {state}: {self.value_table[state]:.2f}")
            # 상태번호 - 해당 상태의 가치 
