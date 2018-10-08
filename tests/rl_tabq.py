import gym
import numpy as np
import unittest

from rl.algorithms.tabular_q import TabularQ

class TabQTestCase(unittest.TestCase):

    def test_frozen_finds_good_solution(self):
        env = gym.make('FrozenLake-v0')
        agent = TabularQ(env)
        agent.train(20)
        N = 100
        episode_rewards = []
        for episode in range(N):
            episode_reward = 0
            state = env.reset()
            done = False
            while not done:
                state, reward, done, _ = env.step(agent.pick_action(state))
                episode_reward += reward
            episode_rewards.append(episode_reward)
        assert np.mean(episode_rewards) > 0.4
        assert np.max(episode_rewards) == 1.0
