from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from practice.common.random_gridworld import Agent, GridWorld


def main():
    env = GridWorld()
    agent = Agent()
    data = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
    gamma = 1.0
    reward = -1
    alpha = 0.001

    for k in range(50000):
        done = False
        history = []

        while not done:
            action = agent.select_action()
            x, y = env.get_state()
            _, reward, done = env.step(action)
            history.append((x, y, reward))
        env.reset()

        cum_reward = 0
        for transition in history[::-1]:
            #
            #   <실습>
            #
            x, y, r = transition
            cum_reward = r + gamma * cum_reward
            data[x][y] += alpha * (cum_reward - data[x][y])
            
    for row in data:
        print(row)

if __name__ == '__main__':
    main()