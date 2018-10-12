import functools
import gym
from gym.envs.classic_control.pendulum import PendulumEnv, angle_normalize
import numpy as np

import irl_benchmark.utils as utils


class FeatureWrapper(gym.Wrapper):
    '''Provide features in info dictionary.'''

    def __init__(self, env):
        super(FeatureWrapper, self).__init__(env)

    def reset(self):
        '''Reset environment and return initial state.
        No changes to base class reset function.
        '''
        self.current_state = self.env.reset()
        return self.current_state

    def step(self, action):
        '''Call base class step method but also log features.

        Args:
          action: `int` corresponding to action to take
        Returns:
          next_state: from env.observation_space, via base class
          reward: `float`, via base class
          terminated: `bool`, via base class
          info: `dictionary` w/ additional key 'features' compared to
            base class, provided by this class's features method
        '''
        # execute action
        next_state, reward, terminated, info = self.env.step(action)

        info['features'] = self.features(self.current_state, action,
                                         next_state)

        # remember which state we are in:
        self.current_state = next_state

        return next_state, reward, terminated, info

    def features(self, current_state, action, next_state):
        '''Return features to be saved in step method's info dictionary.'''
        raise NotImplementedError()

    def feature_shape(self):
        '''Get shape of features.'''
        raise NotImplementedError()

    def feature_range(self):
        '''Get maximum and minimum values of all k features.

        Returns:
        `np.ndarray` of shape (2, k) w/ max in 1st and min in 2nd row.
        '''
        raise NotImplementedError()


class FrozenLakeFeatureWrapper(FeatureWrapper):
    '''Feature wrapper that was ad hoc written for the FrozenLake env.

    Would also work to get one-hot features for any other discrete env
    such that feature-based algorithms can be used in a tabular setting.
    '''

    def features(self, current_state, action, next_state):
        '''Return one-hot encoding of next_state.'''
        return utils.utils.to_one_hot(next_state, self.env.observation_space.n)

    def feature_shape(self):
        '''Return dimension of the one-hot vectors used as features.'''
        return (self.env.observation_space.n, )

    def feature_range(self):
        '''Get maximum and minimum values of all k features.

        Returns:
        `np.ndarray` of shape (2, k) w/ max in 1st and min in 2nd row.
        '''
        ranges = np.zeros((2, self.feature_shape()[0]))
        ranges[0] = np.ones(self.feature_shape()[0])
        return ranges


class PendulumFeatureWrapper(FeatureWrapper):
    def features(self, current_state, action, next_state):
        th, thdot = utils.utils.unwrap_env(self.env.env, PendulumEnv).state
        action = np.clip(action, -self.env.env.max_torque, self.env.env.max_torque)[0]
        return np.array([angle_normalize(th)**2, thdot**2, action**2])

    def feature_shape(self):
        '''Return one-hot encoding of next_state.'''
        return (3, )

# # # # # #
# MAKE FEATURE WRAPPERS ACCESSIBLE BELOW:
# # # # # #

_feature_wrappers = {}

def feature_wrappable_envs():
    return set(_feature_wrappers.keys())

def make(key):
    '''Return a feature wrapper around the environment specified in key.'''
    return _feature_wrappers[key]()


def _register_feature_wrapper(key):
    '''Unified way of wrapping a feature wrapper around a gym environment'''
    def decorator(f):
        @functools.wraps(f)
        def new_f():
            return f(gym.make(key))
        new_f.__doc__ = "Creates feature wrapper for {}".format(key)
        _feature_wrappers[key] = new_f
        return new_f
    return decorator

@_register_feature_wrapper('FrozenLake-v0')
def frozen_lake(env):
    return FrozenLakeFeatureWrapper(env)

@_register_feature_wrapper('FrozenLake8x8-v0')
def frozen_lake_8_8(env):
    # same feature wrapper as for 'FrozenLake-v0' can be used
    # as size of state space is automatically extracted
    return FrozenLakeFeatureWrapper(env)

@_register_feature_wrapper('Pendulum-v0')
def pendulum(env):
    return PendulumFeatureWrapper(env)

