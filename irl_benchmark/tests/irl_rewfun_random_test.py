from irl_benchmark.envs import make_env, make_wrapped_env
from irl_benchmark.irl.reward.reward_function import TabularRewardFunction, FeatureBasedRewardFunction


def test_random_tabular_function():
    env = make_env('FrozenLake-v0')
    rf = TabularRewardFunction(env, 'random')


def test_random_featb_function():
    env = make_wrapped_env('FrozenLake-v0', with_feature_wrapper=True)
    rf = FeatureBasedRewardFunction(env, 'random')


def test_maze():
    env = make_wrapped_env('FrozenLake-v0', with_feature_wrapper=True)
    rf = FeatureBasedRewardFunction(env, 'random')
