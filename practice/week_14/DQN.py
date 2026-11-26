from argparse import ArgumentParser
from collections import deque
import random
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        self.buffer.append(data)

    def __len__(self):
        return len(self.buffer)

    def get_batch(self):
        data = random.sample(self.buffer, self.batch_size)

        state = torch.tensor(np.stack([x[0] for x in data]))
        action = torch.tensor(np.array([x[1] for x in data]).astype(np.int64))
        reward = torch.tensor(np.array([x[2] for x in data]).astype(np.float32))
        next_state = torch.tensor(np.stack([x[3] for x in data]))
        done = torch.tensor(np.array([x[4] for x in data]).astype(np.int32))
        return state, action, reward, next_state, done


class QNet(nn.Module):
    def __init__(self, action_size):
        super().__init__()
        self.l1 = nn.Linear(4, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


class DQNAgent:
    def __init__(self):
        self.gamma = 0.98
        self.lr = 0.0005
        self.epsilon = 0.1
        self.buffer_size = 10000
        self.batch_size = 32
        self.action_size = 2

        self.replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size)
        self.qnet = QNet(self.action_size)
        self.qnet_target = QNet(self.action_size)
        self.sync_qnet()
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=self.lr)

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            state = torch.tensor(state[np.newaxis, :])
            qs = self.qnet(state)
            return qs.argmax().item()

    def update(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)
        if len(self.replay_buffer) < self.batch_size:
            return

        # 실습 구현
        # 미니배치 샘플링
        states, actions, rewards, next_states, dones = self.replay_buffer.get_batch()

        # Q(s, a) 계산
        q_values = self.qnet(states)
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # max Q'(s', a') 계산 (타깃 네트워크로부터)
        next_q_values = self.qnet_target(next_states)
        next_q_value = next_q_values.max(1)[0]

        # done이면 다음 상태의 Q값은 0
        target = rewards + self.gamma * next_q_value * (1 - dones)

        # 손실 계산 (MSE)
        loss = F.mse_loss(q_value, target.detach())

        # 역전파 및 최적화
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()        
        
    def sync_qnet(self):
        self.qnet_target.load_state_dict(self.qnet.state_dict())


def main():
    parser = ArgumentParser(description="Train and evaluate DQN on CartPole.")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    sync_interval = 20
    agent = DQNAgent()
    with gym.make("CartPole-v1") as env:
        for episode in range(args.episodes):
            state, _ = env.reset()
            done = False
            total_reward = 0

            while not done:
                action = agent.get_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                agent.update(state, action, reward, next_state, terminated)
                state = next_state
                total_reward += reward

            if (episode + 1) % sync_interval == 0:
                agent.sync_qnet()
            if episode % 10 == 0:
                print(f"Episode {episode + 1}, total reward: {total_reward}")

    agent.epsilon = 0.0
    with gym.make("CartPole-v1", render_mode=None if args.headless else "human") as env:
        for episode in range(args.eval_episodes):
            state, _ = env.reset()
            done = False
            total_reward = 0
            while not done:
                state, reward, terminated, truncated, _ = env.step(agent.get_action(state))
                done = terminated or truncated
                total_reward += reward
            print(f"Evaluation {episode + 1}, total reward: {total_reward}")


if __name__ == "__main__":
    main()
