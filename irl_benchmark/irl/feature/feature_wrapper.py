"""Module for feature wrappers providing features for different environments."""

from abc import abstractmethod
import functools
from typing import Union, Tuple

import gym
import numpy as np

from irl_benchmark.utils.general import to_one_hot


class FeatureWrapper(gym.Wrapper):
    """Wrapper that adds features to the info dictionary in the step function.

    Generally, each environment needs its own feature wrapper."""

    def __init__(self, env: gym.Env):
        """

        Parameters
        ----------
        env: gym.Env
            The gym environment to be wrapped.
        """
        super(FeatureWrapper, self).__init__(env)
        self.current_state = None

    def reset(self, **kwargs):  # pylint: disable=E0202, R0801
        """ Reset environment and return initial state. No changes to base class reset function."""
        self.current_state = self.env.reset()
        return self.current_state

    def step(self, action: Union[np.ndarray, int, float]
             ) -> Tuple[Union[np.ndarray, float, int], float, bool, dict]:
        """

        Parameters
        ----------
        action: Union[np.ndarray, int, float]

        Returns
        -------
        Tuple[Union[np.ndarray, float, int], float, bool, dict]
            Tuple with values for (state, reward, done, info).
            Normal return values of any gym step function.
            A field 'features' is added to the returned info dict.
        """
        # pylint: disable=E0202

        # execute action:
        next_state, reward, terminated, info = self.env.step(action)

        info['features'] = self.features(self.current_state, action,
                                         next_state)

        # remember which state we are in:
        self.current_state = next_state

        return next_state, reward, terminated, info

    @abstractmethod
    def features(self, current_state: Union[np.ndarray, int, float],
                 action: Union[np.ndarray, int, float],
                 next_state: Union[np.ndarray, int, float]) -> np.ndarray:
        """Return features to be saved in step method's info dictionary."""
        raise NotImplementedError()

    @abstractmethod
    def feature_shape(self) -> tuple:
        """Get shape of features."""
        raise NotImplementedError()

    @abstractmethod
    def feature_range(self) -> np.ndarray:
        """Get minimum and maximum values of all k features.

        Returns
        -------
        np.ndarray
            The minimum and maximum values in an array of shape (2, k).
            First row corresponds to minimum values and second row to maximum values.
        """
        raise NotImplementedError()

    @abstractmethod
    def feature_array(self) -> np.ndarray:
        """ Get features for the entire domain as an array.
        Has to be overwritten in each feature wrapper.

        Returns
        -------
        np.ndarray
            The features for the entire domain as an array.
            Shape: (domain_size, feature_size).
        """
        raise NotImplementedError()


class FrozenLakeFeatureWrapper(FeatureWrapper):
    """Feature wrapper that was ad hoc written for the FrozenLake env.

    Would also work to get one-hot features for any other discrete env
    such that feature-based algorithms can be used in a tabular setting.
    """

    def features(self, current_state, action, next_state):
        '''Return one-hot encoding of next_state.'''
        return to_one_hot(next_state, self.env.observation_space.n)

    def feature_shape(self):
        '''Return dimension of the one-hot vectors used as features.'''
        return (self.env.observation_space.n, )

    def feature_range(self):
        '''Get maximum and minimum values of all k features.

        Returns:
        `np.ndarray` of shape (2, k) w/ max in 1st and min in 2nd row.
        '''
        ranges = np.zeros((2, self.feature_shape()[0]))
        ranges[1, :] = 1.0
        return ranges

    def feature_array(self):
        """Returns feature array for FrozenLake. Each state in the domain
        corresponds to a one_hot vector. Features of all states together
        are the identity matrix."""
        return np.eye(self.env.observation_space.n)


# # # # # #
# MAKE FEATURE WRAPPERS ACCESSIBLE BELOW:
# # # # # #

_FEATURE_WRAPPERS = {}


def feature_wrappable_envs() -> set:
    """Return list of ids for all gym environments that can currently be
    wrapped with a feature wrapper."""
    return set(_FEATURE_WRAPPERS.keys())


def make(key: str) -> FeatureWrapper:
    """Return a feature wrapper around the gym environment specified with key.

    Parameters
    ----------
    key: str
        A gym environment's id (can be found as env.spec.id),
        for example 'FrozenLake-v0'.

    Returns
    -------
    FeatureWrapper
        An environment created as gym.make(key) wrapped in an
        adequate feature wrapper.
    """
    # the _FEATURE_WRAPPERS dict is filled below by registering environments with
    # @_register_feature_wrapper.
    return _FEATURE_WRAPPERS[key]()


def _register_feature_wrapper(key: str):
    """Unified way of registering feature wrappers for gym environments."""

    def decorator(decorated_function):
        @functools.wraps(decorated_function)
        def wrapper_factory():
            # return a new feature wrapper around a new gym environment:
            return decorated_function(gym.make(key))

        # add docstring
        wrapper_factory.__doc__ = "Creates feature wrapper for {}".format(key)
        # add to list of feature-wrappable environments
        _FEATURE_WRAPPERS[key] = wrapper_factory
        return wrapper_factory

    return decorator


@_register_feature_wrapper('FrozenLake-v0')
def frozen_lake(env: gym.Env):
    """Register 'FrozenLake-v0' feature wrapper."""
    return FrozenLakeFeatureWrapper(env)


@_register_feature_wrapper('FrozenLake8x8-v0')
def frozen_lake_8_8(env: gym.Env):
    """Register 'FrozenLake-v2' feature wrapper."""
    # same feature wrapper as for 'FrozenLake-v0' can be used
    # as size of state space is automatically extracted
    return FrozenLakeFeatureWrapper(env)
