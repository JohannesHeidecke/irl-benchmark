"""Module for feature wrappers providing features for different environments."""

import functools
from abc import abstractmethod
from typing import Union, Tuple

import gym
import numpy as np

# from irl_benchmark.envs.maze_world import MazeWorld, RANDOM_QUIT_CHANCE, REWARD_SMALL, REWARD_MEDIUM, REWARD_LARGE
from irl_benchmark.utils.general import to_one_hot
from irl_benchmark.utils.wrapper import unwrap_env


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

    def reset(self, **kwargs):  # pylint: disable=method-hidden, R0801
        """ Reset environment and return initial state.
        No changes to base class reset function."""
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
            The value of 'features' is a np.ndarray of shape (1, d)
            where d is the dimensionality of the feature space.
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
        """Return features for a single state or a state-action pair or a
        state-action-next_state triplet. To be saved in step method's info dictionary.

        Parameters
        ----------
        current_state: Union[np.ndarray, int, float]
            The current state. Can be None if used reward function has properties
            action_in_domain == False and next_state_in_domain == False and if
            next_state is not None. In that case the features are calculated for
            the next state and used for the reward function R(s) - the reward for
            reaching next_state.
        action: Union[np.ndarray, int, float]
            A single action. Has to be given if used reward function has property
            action_in_domain == True.
        next_state: Union[np.ndarray, int, float]
            The next state. Has to be given if used reward function has property
            next_state_in_domain == True.

        Returns
        -------
        np.ndarray
            The features in a numpy array of shape (1, d), where d is the
            dimensionality of the feature space (see :meth:`.feature_dimensionality`).
        """
        raise NotImplementedError()

    @abstractmethod
    def feature_dimensionality(self) -> tuple:
        """Get the dimensionality of the feature space."""
        raise NotImplementedError()

    @abstractmethod
    def feature_range(self) -> np.ndarray:
        """Get minimum and maximum values of all d features, where d is the
        dimensionality of the feature space (see :meth:`.feature_dimensionality`)

        Returns
        -------
        np.ndarray
            The minimum and maximum values in an array of shape (2, d).
            First row corresponds to minimum values and second row to maximum values.
        """
        raise NotImplementedError()

    @abstractmethod
    def feature_array(self) -> np.ndarray:
        """ Get features for the entire domain as an array.
        Has to be overwritten in each feature wrapper.
        Wrappers for large environments will not implement this method.

        Returns
        -------
        np.ndarray
            The features for the entire domain as an array.
            Shape: (domain_size, d).
        """
        raise NotImplementedError()


class FrozenLakeFeatureWrapper(FeatureWrapper):
    """Feature wrapper that was ad hoc written for the FrozenLake env.

    Would also work to get one-hot features for any other discrete env
    such that feature-based IRL algorithms can be used in a tabular setting.
    """

    def features(self, current_state: None, action: None,
                 next_state: int) -> np.ndarray:
        """Return features to be saved in step method's info dictionary.
        One-hot encoding the next state.

        Parameters
        ----------
        current_state: None
        action: None
        next_state: int
            The next state.

        Returns
        -------
        np.ndarray
            The features in a numpy array.
        """
        assert next_state is not None
        if isinstance(next_state, (int, np.int64, np.ndarray)):
            return to_one_hot(next_state, self.env.observation_space.n)
        else:
            raise NotImplementedError()

    def feature_dimensionality(self) -> Tuple:
        """Return dimension of the one-hot vectors used as features."""
        return (self.env.observation_space.n, )

    def feature_range(self):
        """Get maximum and minimum values of all k features.

        Returns:
        `np.ndarray` of shape (2, k) w/ max in 1st and min in 2nd row.
        """
        ranges = np.zeros((2, self.feature_dimensionality()[0]))
        ranges[1, :] = 1.0
        return ranges

    def feature_array(self):
        """Returns feature array for FrozenLake. Each state in the domain
        corresponds to a one_hot vector. Features of all states together
        are the identity matrix."""
        return np.eye(self.env.observation_space.n)


class MazeWorldFeatureWrapper(FeatureWrapper):
    def features(self, current_state: np.ndarray, action: int,
                 next_state: None) -> np.ndarray:
        """Return features to be saved in step method's info dictionary.

        There are four feature variables: expected walking distance,
        probability of reaching a small reward field, probability of reaching
        a medium reward field, probability of reaching a large reward field.
        Only one of the last three values will be non-zero."""

        maze_env = unwrap_env(self.env, MazeWorld)

        # can only calculate features for a single state-action pair.
        assert len(current_state.shape) == 1

        # special case: not at any position:
        if np.sum(current_state[:maze_env.num_rewards]) == 0:
            return np.array([1, 0, 0, 0])
        path_len = maze_env.get_path_len(current_state, action)

        # special case: all rewards collected:
        if np.sum(current_state[maze_env.num_rewards:]) == 0:
            return np.zeros(4)

        assert path_len > 0
        # special case: walking to current position
        if path_len == 1:
            # assert that agent is walking to its current position:
            assert current_state[action] == 1.0
            expected_walking_distance = 1.0
        else:
            # calculate expected walking distance feature:
            possible_distances = np.arange(1, path_len)
            prob_getting_to_distance = (
                1 - RANDOM_QUIT_CHANCE)**possible_distances
            prob_stopping_at_distance = np.ones_like(
                possible_distances, dtype=np.float32)
            prob_stopping_at_distance[:-1] = RANDOM_QUIT_CHANCE
            expected_walking_distance = np.sum(
                possible_distances * prob_getting_to_distance *
                prob_stopping_at_distance)

        # coin collection probabilities:
        ccps = np.zeros(3)
        rew_value = maze_env.get_rew_value(current_state, action)
        if rew_value != 0.:
            assert rew_value in [REWARD_SMALL, REWARD_MEDIUM, REWARD_LARGE]
            rew_value_index = [REWARD_SMALL, REWARD_MEDIUM,
                               REWARD_LARGE].index(rew_value)
            if path_len == 1:
                ccps[rew_value_index] = (1 - RANDOM_QUIT_CHANCE)
            else:
                ccps[rew_value_index] = (1 - RANDOM_QUIT_CHANCE)**(
                    path_len - 1)

        return np.concatenate((np.array([expected_walking_distance]), ccps))

    def feature_dimensionality(self):
        """Return dimension of the one-hot vectors used as features."""
        return (4, )

    def feature_range(self):
        """Return minimum and maximum values of features.
        Max is set to an arbitrary high value."""
        return np.array([[0, 0, 0, 0], [1e3] * 4])

    def feature_array(self) -> np.ndarray:
        """ Get features for the entire domain as an array.
        Has to be overwritten in each feature wrapper.
        Wrappers for large environments will not implement this method.

        Returns
        -------
        np.ndarray
            The features for the entire domain as an array.
            Shape: (domain_size, d).
        """
        maze_world = unwrap_env(self.env, MazeWorld)
        num_rewards = maze_world.num_rewards
        n_states = num_rewards * 2**num_rewards
        feature_array = np.zeros((n_states, num_rewards, 4))
        for s in range(n_states):
            for a in range(num_rewards):
                state = maze_world.index_to_state(s)
                feature = self.features(state, a, None)
                feature_array[s, a, :] = feature
        return feature_array


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
        An environment created as :func:`irl_benchmark.envs.make_env`(key) wrapped in an
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
            # import unified way of creating environments
            # (usually using gym.make, with some exceptions
            from irl_benchmark.envs import make_env
            # return a new feature wrapper around a new gym environment:
            return decorated_function(make_env(key))

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


@_register_feature_wrapper('MazeWorld0-v0')
def maze_world_0(env: gym.Env):
    """Register MazeWorld0 (10 rewards) feature wrapper."""
    return MazeWorldFeatureWrapper(env)


@_register_feature_wrapper('MazeWorld1-v0')
def maze_world_1(env: gym.Env):
    """Register MazeWorld1 (10 rewards) feature wrapper."""
    return MazeWorldFeatureWrapper(env)
