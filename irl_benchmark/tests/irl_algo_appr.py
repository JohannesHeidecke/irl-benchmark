import gym
import numpy as np
import unittest

from irl_benchmark.irl.algorithms.appr.appr_irl import ApprIRL
from irl_benchmark.irl.feature.feature_wrapper import FrozenLakeFeatureWrapper
from irl_benchmark.irl.collect import collect_trajs
from irl_benchmark.irl.reward.reward_function import FeatureBasedRewardFunction
from irl_benchmark.irl.reward.reward_wrapper import RewardWrapper
from irl_benchmark.rl.algorithms import TabularQ


def run_appr_irl(use_projection, duration):
    store_to = 'data/frozen/expert'
    no_episodes = 1000
    max_steps_per_episode = 100
    env = gym.make('FrozenLake-v0')
    env = FrozenLakeFeatureWrapper(env)
    expert_agent = TabularQ(env)
    expert_agent.train(duration)
    expert_trajs = collect_trajs(env, expert_agent, no_episodes,
                                 max_steps_per_episode, store_to)
    reward_function = FeatureBasedRewardFunction(
        env, np.random.normal(size=16))
    env = RewardWrapper(env, reward_function)
    appr_irl = ApprIRL(env, expert_trajs, TabularQ, proj=use_projection)
    appr_irl.train(duration, duration/2, verbose=False)
    return appr_irl.distances


class ApprIRLTestCase(unittest.TestCase):

    def test_svm(self, duration=2):
        distances = run_appr_irl(False, duration)
        if duration < 5:
            return
        assert len(distances) >= 3
        assert len(distances) <= 10  # unrealistically high
        assert distances[-1] < distances[0]
        assert distances[-1] < 5

    def test_proj(self, duration=2):
        distances = run_appr_irl(True, duration)
        if duration < 5:
            return
        assert len(distances) >= 3
        assert len(distances) <= 10  # unrealistically high
        assert distances[-1] < distances[0]
        assert distances[-1] < 5
