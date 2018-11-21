import gym
import numpy as np

from irl_benchmark.envs import make_wrapped_env
from irl_benchmark.irl.collect import collect_trajs
from irl_benchmark.irl.feature.feature_wrapper import make as feature_make
from irl_benchmark.irl.reward.reward_function import TabularRewardFunction, FeatureBasedRewardFunction
from irl_benchmark.irl.reward.truth import make_true_reward
from irl_benchmark.rl.algorithms.value_iteration import ValueIteration


def test_tabular_function():
    def reward_function_factory(env):
        params = np.zeros(64)
        params[-1] = 1.
        return TabularRewardFunction(env, params)

    env = make_wrapped_env(
        'FrozenLake8x8-v0',
        reward_function_factory=reward_function_factory,
        with_model_wrapper=True)
    agent = ValueIteration(env)
    agent.train(1)
    trajs = collect_trajs(env, agent, 10)
    for traj in trajs:
        for i in range(len(traj['rewards'])):
            assert np.isclose(traj['rewards'][i], traj['true_rewards'][i])


def test_featb_function():
    def reward_function_factory(env):
        params = np.zeros(64)
        params[-1] = 1.
        return FeatureBasedRewardFunction(env, params)

    env = make_wrapped_env(
        'FrozenLake8x8-v0',
        with_feature_wrapper=True,
        reward_function_factory=reward_function_factory,
        with_model_wrapper=True)
    agent = ValueIteration(env)
    agent.train(1)
    trajs = collect_trajs(env, agent, 10)
    for traj in trajs:
        for i in range(len(traj['rewards'])):
            assert np.isclose(traj['rewards'][i], traj['true_rewards'][i])


def test_tab_featb_functions():
    env = feature_make('FrozenLake8x8-v0')
    params = np.zeros(64)
    params[-1] = 1.
    rf = FeatureBasedRewardFunction(env, params)
    domain = rf.domain()
    rf2 = TabularRewardFunction(env, params)
    rf_true = make_true_reward('FrozenLake8x8-v0')
    rew1 = rf.reward(domain)
    rew2 = rf2.reward(domain)
    rew_true = rf_true.reward(domain)
    assert np.all(rew_true == rew1)
    assert np.all(rew1 == rew2)
    assert rew_true.shape == rew1.shape
    assert rew1.shape == rew2.shape
