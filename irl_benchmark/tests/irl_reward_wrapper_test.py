import numpy as np

from irl_benchmark.envs import make_wrapped_env
from irl_benchmark.irl.reward.reward_function import FeatureBasedRewardFunction, TabularRewardFunction
from irl_benchmark.irl.reward.reward_wrapper import RewardWrapper
from irl_benchmark.utils.wrapper import unwrap_env


def test_update_parameters_frozen_feature():
    def rew_fun_factory(env):
        return FeatureBasedRewardFunction(env, 'random')

    env = make_wrapped_env(
        'FrozenLake-v0',
        with_feature_wrapper=True,
        reward_function_factory=rew_fun_factory)

    reward_wrapper = unwrap_env(env, RewardWrapper)
    params = np.copy(reward_wrapper.reward_function.parameters)
    domain = reward_wrapper.reward_function.domain()
    rews = reward_wrapper.reward_function.reward(domain)
    reward_wrapper.update_reward_parameters(2 * params)
    rews2 = reward_wrapper.reward_function.reward(domain)
    assert np.all(np.isclose(2 * rews, rews2))
    reward_wrapper.update_reward_parameters(np.zeros_like(params))
    rews3 = reward_wrapper.reward_function.reward(domain)
    assert np.all(np.isclose(rews3, np.zeros_like(rews3)))


def test_update_parameters_frozen_tabular():
    def rew_fun_factory(env):
        return TabularRewardFunction(env, 'random')

    env = make_wrapped_env(
        'FrozenLake-v0',
        with_feature_wrapper=False,
        reward_function_factory=rew_fun_factory)

    reward_wrapper = unwrap_env(env, RewardWrapper)
    params = np.copy(reward_wrapper.reward_function.parameters)
    domain = reward_wrapper.reward_function.domain()
    rews = reward_wrapper.reward_function.reward(domain)
    reward_wrapper.update_reward_parameters(2 * params)
    rews2 = reward_wrapper.reward_function.reward(domain)
    assert np.all(np.isclose(2 * rews, rews2))
    reward_wrapper.update_reward_parameters(np.zeros_like(params))
    rews3 = reward_wrapper.reward_function.reward(domain)
    assert np.all(np.isclose(rews3, np.zeros_like(rews3)))
