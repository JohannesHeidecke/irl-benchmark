import gym
from gym.envs.toy_text.discrete import DiscreteEnv
from gym.wrappers.time_limit import TimeLimit

from irl_benchmark.envs import make_env
from irl_benchmark.irl.feature.feature_wrapper import FrozenLakeFeatureWrapper
from irl_benchmark.irl.feature import feature_wrapper
from irl_benchmark.irl.reward.reward_function import FeatureBasedRewardFunction
from irl_benchmark.irl.reward.reward_wrapper import RewardWrapper
from irl_benchmark.utils.wrapper import is_unwrappable_to, unwrap_env


def test_unwrap():
    env = make_env('FrozenLake-v0')
    assert env.env is unwrap_env(env, DiscreteEnv)

    # No unwrapping needed:
    assert env is unwrap_env(env, gym.Env)

    # Unwrap all the way:
    assert env.env is unwrap_env(env)

    env = FrozenLakeFeatureWrapper(env)
    assert env.env.env is unwrap_env(env, DiscreteEnv)

    # No unwrapping needed:
    assert env is unwrap_env(env, FrozenLakeFeatureWrapper)

    # Unwrap all the way:
    assert env.env.env is unwrap_env(env)

    # check types:
    assert isinstance(unwrap_env(env, DiscreteEnv), DiscreteEnv)
    assert isinstance(
        unwrap_env(env, feature_wrapper.FeatureWrapper),
        feature_wrapper.FeatureWrapper)
    assert isinstance(
        unwrap_env(env, FrozenLakeFeatureWrapper), FrozenLakeFeatureWrapper)
    assert isinstance(
        unwrap_env(env, FrozenLakeFeatureWrapper),
        feature_wrapper.FeatureWrapper)
    assert isinstance(unwrap_env(env), gym.Env)


def test_is_unwrappable_to():
    assert is_unwrappable_to(make_env('FrozenLake-v0'), TimeLimit)
    assert is_unwrappable_to(make_env('FrozenLake-v0'), DiscreteEnv)
    assert is_unwrappable_to(
        feature_wrapper.make('FrozenLake-v0'), FrozenLakeFeatureWrapper)
    assert is_unwrappable_to(
        feature_wrapper.make('FrozenLake8x8-v0'), FrozenLakeFeatureWrapper)
    assert is_unwrappable_to(
        feature_wrapper.make('FrozenLake-v0'), feature_wrapper.FeatureWrapper)
    env = feature_wrapper.make('FrozenLake-v0')
    reward_function = FeatureBasedRewardFunction(env, 'random')
    env = RewardWrapper(env, reward_function)
    assert is_unwrappable_to(env, RewardWrapper)
    assert is_unwrappable_to(env, feature_wrapper.FeatureWrapper)
    assert is_unwrappable_to(env, DiscreteEnv)
    assert is_unwrappable_to(env, gym.Env)
