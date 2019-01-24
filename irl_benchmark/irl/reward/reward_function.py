"""Module containing reward functions to be used for IRL."""
from abc import ABC, abstractmethod
from copy import copy
from typing import NamedTuple, Union

import gym
import numpy as np
from gym.envs.toy_text.discrete import DiscreteEnv
from gym.spaces.discrete import Discrete as DiscreteSpace

import irl_benchmark.utils as utils
# from irl_benchmark.envs.maze_world import MazeWorld
from irl_benchmark.irl.feature.feature_wrapper import FeatureWrapper


# Define some custom named tuples for easier handling
# of the reward function domains:
class State(NamedTuple):
    """A tuple containing an array of states."""
    state: np.ndarray


class StateAction(NamedTuple):
    """A tuple containing an array of states and an array of actions."""
    state: np.ndarray
    action: np.ndarray


class StateActionState(NamedTuple):
    """A tuple containing an array each for states, actions, and next states."""
    state: np.ndarray
    action: np.ndarray
    next_state: np.ndarray


class BaseRewardFunction(ABC):
    """The base class for all reward functions."""

    def __init__(self,
                 env: gym.Env,
                 parameters: Union[None, str, np.ndarray] = None,
                 action_in_domain: bool = False,
                 next_state_in_domain: bool = False):
        """ The abstract base class for reward functions

        Parameters
        ----------
        env: gym.Env
            A gym environment for which the reward function is defined.
        parameters: Union[None, str, np.ndarray]
            A numpy ndarray containing the parameters. If value is 'random',
            initializes with random parameters (mean 0, standard deviation 1).
        action_in_domain: bool
            Indicates whether actions are in the domain, i.e. R(s, a) or R(s, a, s')
        next_state_in_domain: bool
            Indicates whether next states are in the domain, i.e. R(s, a, s')
        """
        self.env = env
        self.action_in_domain = action_in_domain
        if next_state_in_domain:
            assert action_in_domain
        self.next_state_in_domain = next_state_in_domain
        self.parameters = parameters

    def domain(self) -> Union[State, StateAction, StateActionState]:
        """Return the entire domain of the reward function.

        Returns
        -------
        Union[State, StateAction, StateActionState]
            The domain of the reward function.
        """
        if utils.wrapper.is_unwrappable_to(self.env, DiscreteEnv):
            n_states = self.env.observation_space.n
            n_actions = self.env.action_space.n
            states = np.arange(n_states)
        elif utils.wrapper.is_unwrappable_to(self.env, MazeWorld):
            maze_env = utils.wrapper.unwrap_env(self.env, MazeWorld)
            num_rewards = maze_env.num_rewards
            n_states = num_rewards * 2**num_rewards
            n_actions = num_rewards
            states = np.array(
                [maze_env.index_to_state(i) for i in range(n_states)])
        else:
            raise NotImplementedError()

        if self.action_in_domain:
            # if domain contains actions: extend domain
            states = np.repeat(states, n_actions, axis=0)
            actions = np.arange(n_actions)
            actions = np.tile(actions, n_states)
            if self.next_state_in_domain:
                # if domain contains next states: extend domain
                states = np.repeat(states, n_states)
                actions = np.repeat(actions, n_states)
                next_states = np.arange(n_states)
                next_states = np.tile(next_states, n_states * n_actions)
                # return the adequate namedtuple:
                return StateActionState(states, actions, next_states)
            return StateAction(states, actions)
        return State(states)

    def domain_sample(self, batch_size: int
                      ) -> Union[State, StateAction, StateActionState]:
        """Return a batch sampled from the domain of size batch_size.

        Parameters
        ----------
        batch_size: int
            The size of the batch.

        Returns
        -------
        Union[State, StateAction, StateActionState]
            A batch sampled from the domain of the reward function.
        """
        raise NotImplementedError()

    @abstractmethod
    def reward(self, domain_batch: Union[State, StateAction, StateActionState]
               ) -> np.ndarray:
        """Return rewards for a domain batch.

        Parameters
        ----------
        domain_batch: Union[State, StateAction, StateActionState, np.ndarray]
            A batch of the domain (can in principle also be the entire domain).
            See :func:`domain` and :func:`domain_sample`.

        Returns
        -------
        np.ndarray
            Rewards for the given domain batch.
            The rewards are of shape (batch_size,)
        """
        raise NotImplementedError()

    def update_parameters(self, parameters: np.ndarray):
        """Update the parameters of the reward function.

        Parameters
        ----------
        parameters: np.ndarray
            The new parameters to be used in the reward function.
        """
        self.parameters = parameters


class TabularRewardFunction(BaseRewardFunction):
    """Rewards for each possible input are stored in a table.

    Only suitable for relatively small environments.
    The self.parameters in this case are the reward table's values.
    """

    def __init__(self,
                 env: gym.Env,
                 parameters: Union[None, str, np.ndarray] = None,
                 action_in_domain: bool = False,
                 next_state_in_domain: bool = False):
        """

        Parameters
        ----------
        env: gym.Env
            A gym environment for which the reward function is defined.
        parameters: Union[None, str, np.ndarray]
            A numpy ndarray containing the values for all elements in the reward table.
            If value is 'random', initializes with random parameters (mean 0, standard deviation 1).
            The size of parameters must correspond to the size of the domain
            (one table value for each possible input)
        action_in_domain: bool
            Indicates whether actions are in the domain, i.e. R(s, a) or R(s, a, s')
        next_state_in_domain: bool
            Indicates whether next states are in the domain, i.e. R(s, a, s')
        """
        super(TabularRewardFunction, self).__init__(
            env, parameters, action_in_domain, next_state_in_domain)

        # this reward function is only implemented for
        # discrete state and action spaces
        assert isinstance(env.observation_space, DiscreteSpace)
        assert isinstance(env.action_space, DiscreteSpace)
        # calculate number of elements in domain:
        self.domain_size = self.env.observation_space.n
        if self.action_in_domain:
            self.domain_size *= self.env.action_space.n
        if self.next_state_in_domain:
            self.domain_size *= self.env.observation_space.n

        if parameters == 'random':
            self.parameters = np.random.standard_normal(size=self.domain_size)
        else:
            self.parameters = np.array(parameters)
        assert len(self.parameters) == self.domain_size

    def domain_to_index(
            self, domain_batch: Union[State, StateAction, StateActionState]
    ) -> np.ndarray:
        """Convert a domain batch into  an numpy ndarray of reward table indices.

        Parameters
        ----------
        domain_batch: Union[State, StateAction, StateActionState]
            A domain batch. Can also be the entire domain.

        Returns
        -------
        np.ndarray
            A numpy array of corresponding reward table indices.
        """
        assert utils.wrapper.is_unwrappable_to(self.env, DiscreteEnv)
        index = copy(domain_batch.state)
        if self.action_in_domain:
            index *= self.env.action_space.n
            index += domain_batch.action
            if self.next_state_in_domain:
                index *= self.env.observation_space.n
                index += domain_batch.next_state
        return index

    def reward(self, domain_batch: Union[State, StateAction, StateActionState]
               ) -> np.ndarray:
        """Return the corresponding rewards of a domain_batch.

        Parameters
        ----------
        domain_batch: Union[State, StateAction, StateActionState]
            A domain batch. Can also be the entire domain.

        Returns
        -------
        np.ndarray
            The rewards for a domain batch.
        """
        indices = self.domain_to_index(domain_batch)
        return self.parameters[indices]

    def domain_sample(self, batch_size: int
                      ) -> Union[State, StateAction, StateActionState]:
        """Return a batch sampled from the domain of size batch_size.

        Parameters
        ----------
        batch_size: int
            The size of the batch.

        Returns
        -------
        Union[State, StateAction, StateActionState]
            A batch sampled from the domain of the reward function.
        """
        raise NotImplementedError()


class FeatureBasedRewardFunction(BaseRewardFunction):
    """A reward function which is linear in some provided features.

    The self.parameters are the coefficients that are multiplied with
    the features to get the reward (standard inner product).
    """

    def __init__(self,
                 env: gym.Env,
                 parameters: Union[None, str, np.ndarray] = None,
                 action_in_domain: bool = False,
                 next_state_in_domain: bool = False):
        """

        Parameters
        ----------
        env: gym.Env
            A gym environment. The environment has to be wrapped in a FeatureWrapper.
        parameters: Union[None, str, np.ndarray]
            The parameters of the reward function. One parameter for each feature.
            If value is 'random', initializes with random parameters (mean 0, standard deviation 1).
        """

        assert utils.wrapper.is_unwrappable_to(env, FeatureWrapper)
        super(FeatureBasedRewardFunction, self).__init__(
            env, parameters, action_in_domain, next_state_in_domain)

        if parameters == 'random':
            parameters_shape = utils.wrapper.unwrap_env(
                self.env, FeatureWrapper).feature_dimensionality()
            self.parameters = np.random.standard_normal(parameters_shape)

    def reward_from_features(self, feature_batch: np.ndarray) -> np.ndarray:
        """Return corresponding rewards for a domain batch.

        Parameters
        ----------
        feature_batch: np.ndarray
            An array of features

        Returns
        -------
        np.ndarray
            The rewards for given features.
        """
        reward = np.dot(self.parameters.reshape(1, -1), feature_batch.T)
        return reward

    def reward(self, domain_batch: Union[State, StateAction, StateActionState]
               ) -> np.ndarray:
        """Return rewards for a domain batch.
        Converts to features internally to calculate the reward.

        Parameters
        ----------
        domain_batch: Union[State, StateAction, StateActionState, np.ndarray]
            A batch of the domain (can in principle also be the entire domain).
            See :func:`domain` and :func:`domain_sample`.

        Returns
        -------
        np.ndarray
            Rewards for the given domain batch.
            The rewards are of shape (batch_size,)
        """
        features = self._domain_to_features(domain_batch)
        return self.reward_from_features(features).reshape((-1))

    def _domain_to_features(
            self, domain_batch: Union[State, StateAction, StateActionState]
    ) -> np.ndarray:
        """Convert elements of the domain to features.

        Parameters
        ----------
        domain_batch: Union[State ,StateAction, StateActionState]
            A batch of elements from the reward function's domain.

        Returns
        -------
        np.ndarray
            A numpy array containing features.

        """
        # Get the feature wrapper for the environment:
        feature_wrapper = utils.wrapper.unwrap_env(self.env, FeatureWrapper)

        assert isinstance(domain_batch.state, (np.ndarray, tuple, float, int))

        state = domain_batch.state
        if self.action_in_domain:
            action = domain_batch.action
        else:
            action = None
        if self.next_state_in_domain:
            next_state = domain_batch.next_state
        else:
            next_state = None
        if not self.action_in_domain and not self.next_state_in_domain:
            next_state = state
            state = None

        if isinstance(domain_batch.state, (np.ndarray, tuple)) \
                and len(domain_batch.state.shape) > 0:
            n_inputs = domain_batch.state.shape[0]
        else:
            return feature_wrapper.features(state, action, next_state)

        feature_array = []

        for i in range(n_inputs):

            if state is not None:
                feat_in_state = state[i]
            else:
                feat_in_state = None
            feat_in_action = action[i] if action is not None else None
            feat_in_next_state = next_state[
                i] if next_state is not None else None

            features = feature_wrapper.features(feat_in_state, feat_in_action,
                                                feat_in_next_state)

            feature_array.append(features.reshape(-1))
        return np.array(feature_array)

    def domain_sample(self, batch_size: int
                      ) -> Union[State, StateAction, StateActionState]:
        """Return a batch sampled from the domain of size batch_size.

        Parameters
        ----------
        batch_size: int
            The size of the batch.

        Returns
        -------
        Union[State, StateAction, StateActionState]
            A batch sampled from the domain of the reward function.
        """
        raise NotImplementedError()
