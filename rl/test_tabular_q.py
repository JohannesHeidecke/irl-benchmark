from rl.tabular_q import TabularQ
import gym
import numpy as np


def test_tabular_q_agent():
    env = gym.make('FrozenLake-v0')

    agent = TabularQ(env)
    N = 1000

    sum_rewards = []
    for i in range(N):
        rs = []
        s = env.reset()
        done = False

        while not done:
            a = agent.pick_action(s)
            sp, r, done, info = env.step(a)
            agent.update(s, a, r, sp)
            s = sp
            rs.append(r)

        sum_reward = np.sum(rs)
        sum_rewards.append(sum_reward)
        if i % 10 == 0 and i > 0:
            print({'MA(10) reward': np.mean(sum_rewards[-10:])})

    ma_100_reward = np.mean(sum_rewards[-200:])
    assert ma_100_reward > 0.4
    print(ma_100_reward)


if __name__ == '__main__':
    test_tabular_q_agent()
