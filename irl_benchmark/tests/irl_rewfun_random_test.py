import gym

from irl_benchmark.irl.feature.feature_wrapper import make as feature_make
from irl_benchmark.irl.reward.reward_function import TabularRewardFunction, FeatureBasedRewardFunction


def test_random_tabular_function():
    env = gym.make('FrozenLake-v0')
    rf = TabularRewardFunction(env, 'random')


def test_random_featb_function():
    env = feature_make('FrozenLake-v0')
    rf = FeatureBasedRewardFunction(env, 'random')
