import gym
import numpy as np
import pickle
import unittest
import shutil

from irl.collect import collect_trajs
from irl.feature.feature_wrapper import FrozenFeatureWrapper
from irl.reward.reward_function import FeatureBasedRewardFunction
from irl.reward.reward_wrapper import RewardWrapper
from rl.algorithms.base_algorithm import RandomAgent

class CollectTestCase(unittest.TestCase):

    def test_agents_collect(self):
        env = gym.make('FrozenLake-v0')
        agent = RandomAgent(env)
        trajs = collect_trajs(env, agent, 10, 100, store_to='tests/tempdata/')
        with open('tests/tempdata/trajs.pkl', 'rb') as f:
            pickle_trajs = pickle.load(f)
        for traj in trajs:
            assert type(traj) is dict
            assert len(traj['states']) > 0
            assert len(traj['actions']) > 0
            assert len(traj['rewards']) > 0
            assert len(traj['true_rewards']) == 0
            assert len(traj['features']) == 0
        shutil.rmtree('tests/tempdata/')

    def test_agents_collect_feature_reward(self):
        env = gym.make('FrozenLake-v0')
        feature_wrapper = FrozenFeatureWrapper(env)
        reward_function = FeatureBasedRewardFunction(env, np.random.normal(size=16))
        env = RewardWrapper(feature_wrapper, reward_function)
        agent = RandomAgent(env)
        trajs = collect_trajs(env, agent, 10, 100, store_to='tests/tempdata/')
        with open('tests/tempdata/trajs.pkl', 'rb') as f:
            pickle_trajs = pickle.load(f)
        for traj in trajs:
            assert type(traj) is dict
            assert len(traj['states']) > 0
            assert len(traj['actions']) > 0
            assert len(traj['rewards']) > 0
            assert len(traj['true_rewards']) > 0
            assert len(traj['features']) > 0
        shutil.rmtree('tests/tempdata/')
