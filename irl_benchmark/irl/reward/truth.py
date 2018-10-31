import functools
import gym
import numpy as np

from irl_benchmark.irl.reward.reward_function import TabularRewardFunction

_true_reward_functions = {}


def make(key):
    return _true_reward_functions[key]()


def _register_reward_function(key):
    def decorator(f):
        @functools.wraps(f)
        def new_f():
            return f(gym.make(key))

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
