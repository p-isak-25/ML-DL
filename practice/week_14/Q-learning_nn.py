from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from practice.common.gridworld import GridWorld



# One-hot encoding
def one_hot(state):
    HEIGHT, WIDTH = 3, 4
    vec = np.zeros(HEIGHT * WIDTH, dtype=np.float32)
    y, x = state
    idx = WIDTH * y + x
    vec[idx] = 1.0
    return torch.tensor(vec[np.newaxis, :], dtype=torch.float32)


# Q-network
class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(12, 100)  # Input = one-hot(3x4), hidden = 100
        self.fc2 = nn.Linear(100, 4)   # Output = 4 actions

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Q-learning agent
class QLearningAgent:
    def __init__(self):
        self.gamma = 0.9
        self.lr = 0.01
        self.epsilon = 0.1
        self.action_size = 4

        self.qnet = QNet()
        self.optimizer = optim.SGD(self.qnet.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def get_action(self, state_vec):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            with torch.no_grad():
                qs = self.qnet(state_vec)
                return int(qs.argmax().item())

    def update(self, state, action, reward, next_state, done):
        if done:
            target = torch.tensor([[reward]], dtype=torch.float32)
        else:
            with torch.no_grad():
                next_q = self.qnet(next_state).max(dim=1, keepdim=True)[0]
                target = self.gamma * next_q + reward

        q_values = self.qnet(state)
        q_value = q_values[:, action].unsqueeze(1)  # 선택한 액션 Q 값 가져오기. shape 맞추기

        loss = self.criterion(q_value, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

env = GridWorld()
agent = QLearningAgent()

episodes = 1000
loss_history = []

for episode in range(episodes):
    state = env.reset()
    state = one_hot(state)
    total_loss, cnt = 0, 0
    done = False

    while not done:
        action = agent.get_action(state)
        next_state, reward, done = env.step(action)
        next_state = one_hot(next_state)

        loss = agent.update(state, action, reward, next_state, done)
        total_loss += loss
        cnt += 1
        state = next_state

    average_loss = total_loss / cnt
    loss_history.append(average_loss)


# 에피소드 별 Loss
plt.xlabel('Episode')
plt.ylabel('Loss')
plt.plot(range(len(loss_history)), loss_history)
plt.show()

# Print Q-table
Q = {}
for state in env.states():
    state_vec = one_hot(state)
    with torch.no_grad():
        q_values = agent.qnet(state_vec).squeeze().numpy()
    for action in env.action_space:
        Q[state, action] = float(q_values[action])
env.render_q(Q)
