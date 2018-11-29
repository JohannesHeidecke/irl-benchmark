"""Module containing true reward functions for different environments."""

import functools
import numpy as np

from irl_benchmark.envs import make_wrapped_env
from irl_benchmark.envs.maze_world import REWARD_MOVE, REWARD_SMALL, REWARD_MEDIUM, REWARD_LARGE
from irl_benchmark.irl.reward.reward_function import FeatureBasedRewardFunction, TabularRewardFunction

_true_reward_functions = {}


def make_true_reward(key: str):
    """Make a true reward function."""
    return _true_reward_functions[key]()


def _register_reward_function(key):
    def decorator(f):
        @functools.wraps(f)
        def new_f():
            return f(make_wrapped_env(key, with_feature_wrapper=True))

        new_f.__doc__ = "Creates true reward function for {}".format(key)
        _true_reward_functions[key] = new_f
        return new_f

    return decorator


@_register_reward_function('FrozenLake-v0')
def frozen_lake(env):
    parameters = np.zeros(16)
    parameters[-1] = 1.0
    return TabularRewardFunction(env, parameters)


@_register_reward_function('FrozenLake8x8-v0')
def frozen_lake_8_8(env):
    parameters = np.zeros(64)
    parameters[-1] = 1.0
    return TabularRewardFunction(env, parameters)


@_register_reward_function('MazeWorld0-v0')
def maze_world_0(env):
    parameters = np.array(
        [REWARD_MOVE, REWARD_SMALL, REWARD_MEDIUM, REWARD_LARGE])
    print('Create env for true reward function')
    return FeatureBasedRewardFunction(env, parameters, action_in_domain=True)
