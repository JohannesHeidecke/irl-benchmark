import gym
import numpy as np

from irl_benchmark.irl.reward.reward_function import TabularRewardFunction


true_rewards = {}


key = 'FrozenLake-v0'
parameters = np.zeros(16)
parameters[-1] = 1.0
reward_function = TabularRewardFunction(gym.make(key), parameters)
true_rewards[key] = reward_function


key = 'Pendulum-v0'