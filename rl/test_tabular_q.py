from tabular_q import TabularQ
import gym
import numpy as np

def test_tabular_q():
    env = gym.make('FrozenLake-v0')

    agent = TabularQ(env)
    N = 10000

    avg_rewards = []
    for i in range(N):
        rs = []
        s = env.reset()
        done = False

        while not done:
            a = agent.act(s)
            sp, r, done, info = env.step(a)
            # if r != 0:
            #     print(env.render())
            #     import ipdb;ipdb.set_trace()
            agent.update(s, a, r, sp)
            rs.append(r)
        
        avg_reward = np.mean(rs)
        avg_rewards.append(avg_reward)
        print(avg_reward, end=', ')
    print()
    
if __name__ == '__main__':
    test_tabular_q()