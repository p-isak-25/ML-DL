from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from practice.common.random_gridworld import Agent, GridWorld


def main():
    #TD
    env = GridWorld()
    agent = Agent()
    data = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
    gamma = 1.0
    reward = -1
    alpha = 0.01

    for k in range(50000):
        done = False
        while not done:
            #
            #
            #  <실습>
            #
            #
            x, y = env.get_state()
            action = agent.select_action()
            (nx, ny), reward, done = env.step(action)

            # TD 업데이트
            value = data[x][y]
            next_value = data[nx][ny]
            data[x][y] += alpha * ((reward + gamma * next_value) - value)

        env.reset()
            
    for row in data:
        print(row)

if __name__ == '__main__':
    main()