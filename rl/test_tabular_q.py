from tabular_q import TabularQ
import gym
import numpy as np


def test_tabular_q():
    env = gym.make('FrozenLake-v0')

    agent = TabularQ(env)
    N = 10000

    sum_rewards = []
    for i in range(N):
        rs = []
        s = env.reset()
        done = False

        while not done:
            a = agent.act(s)
            sp, r, done, info = env.step(a)
            agent.update(s, a, r, sp)
            s = sp
            rs.append(r)
        
        sum_reward = np.sum(rs)
        sum_rewards.append(sum_reward)
        if i % 10 == 0 and i > 0:
            print({'MA(10) reward': np.mean(sum_rewards[-10:])})
    print()
    
if __name__ == '__main__':
    test_tabular_q()