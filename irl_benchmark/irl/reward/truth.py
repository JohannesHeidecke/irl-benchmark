import gym
import numpy as np
import functools

from irl_benchmark.irl.reward.reward_function import TabularRewardFunction

_wrapped_envs = {}


def make(key):
    return _wrapped_envs[key]()


def _register_env(key):
    def decorator(f):
        @functools.wraps(f)
        def new_f():
            return f(gym.make(key))
        new_f.__doc__ = "Creates environment {}".format(key)
        _wrapped_envs[key] = new_f
        return new_f
    return decorator


@_register_env('FrozenLake-v0')
def frozen_lake(env):
    parameters = np.zeros(16)
    parameters[-1] = 1.0
    print("Making", env, parameters)
    return TabularRewardFunction(env, parameters)
