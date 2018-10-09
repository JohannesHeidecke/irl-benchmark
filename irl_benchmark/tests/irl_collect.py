import gym
import numpy as np
import pickle
import unittest
import shutil

from irl_benchmark.irl.collect import collect_trajs
from irl_benchmark.irl.feature.feature_wrapper import FrozenLakeFeatureWrapper
from irl_benchmark.irl.reward.reward_function import FeatureBasedRewardFunction
from irl_benchmark.irl.reward.reward_wrapper import RewardWrapper
from irl_benchmark.rl.algorithms.base_algorithm import RandomAgent


class CollectTestCase(unittest.TestCase):
    '''Test if collect module correctly collects trajectories.'''
    def test_agents_collect(self):
        '''Check types and components of returned trajectories.

        Uses a random agent on FrozenLake.'''
        env = gym.make('FrozenLake-v0')
        agent = RandomAgent(env)
        trajs = collect_trajs(env, agent, 10, 100, store_to='tests/tempdata/')
        with open('tests/tempdata/trajs.pkl', 'rb') as f:
            pickle_trajs = pickle.load(f)
        # Return must be a dictionary containing non-empty lists for
        # keys 'states', 'actions', and 'rewards'.  Lists at keys
        # 'true_rewards' and 'features' must be empty as the test uses
        # neither a FeatureWrapper nor a RewardWrapper.
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
        feature_wrapper = FrozenLakeFeatureWrapper(env)
        reward_function = FeatureBasedRewardFunction(
            env, np.random.normal(size=16))
        env = RewardWrapper(feature_wrapper, reward_function)
        agent = RandomAgent(env)
        trajs = collect_trajs(env, agent, 10, 100, store_to='tests/tempdata/')
        with open('tests/tempdata/trajs.pkl', 'rb') as f:
            pickle_trajs = pickle.load(f)
        # This test uses both a FeatureWrapper and a
        # RewardWrapper. Therefore, all components of the return must
        # be non-empty.
        for traj in trajs:
            assert type(traj) is dict
            assert len(traj['states']) > 0
            assert len(traj['actions']) > 0
            assert len(traj['rewards']) > 0
            assert len(traj['true_rewards']) > 0
            assert len(traj['features']) > 0
        shutil.rmtree('tests/tempdata/')
