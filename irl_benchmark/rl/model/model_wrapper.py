"""Module for world model wrappers."""
from typing import Union

import gym
import numpy as np
import sparse


class BaseWorldModelWrapper(gym.Wrapper):
    """The base class for all world model wrappers. A world model wrapper
    provides information about an MDP which are required for model-based
    reinforcement learning methods, such as a matrix of transitions and
    their probabilities."""

    def step(self, action):  #pylint: disable=method-hidden
        """No change to step method, use wrapped environment's method."""
        return self.env.step(action)

    def reset(self, **kwargs):  #pylint: disable=method-hidden
        """No change to reset method, use wrapped environment's method."""
        return self.env.reset(**kwargs)

    def state_to_index(self, state: Union[int, float, np.ndarray]) -> int:
        """Convert a state to a unique index number. To be implemented by
        specific model wrappers for each environment. Opposite of
        :meth:`.index_to_state`.

        Parameters
        ----------
        state: Union[int, float, np.ndarray]
            The state variable, as returned by the wrapped environment.

        Returns
        -------
        int
            The corresponding index of the given state.
        """
        raise NotImplementedError()

    def index_to_state(self, index: int) -> Union[int, float, np.ndarray]:
        """Convert a state index back to a state variable. To be implemented by
        specific model wrappers for each environment. Opposite of
        :meth:`.state_to_index`.

        Parameters
        ----------
        index: int
            The index of a particular state.

        Returns
        -------
        Union[int, float, np.ndarray]
            A state variable, as returned by the wrapped environment.
        """
        raise NotImplementedError()

    def n_states(self) -> int:
        """Return the number of states this environment has."""
        raise NotImplementedError()

    def get_transition_array(self) -> Union[np.ndarray, sparse.COO]:
        """Return the transition matrix for the wrapped environment.

        Returns
        -------
        Union[np.ndarray, sparse.COO]
            The transition array, either a numpy array or a sparse coordinate based
            matrix. Shape (n_states, n_actions, n_states).
        """
        raise NotImplementedError()

    def get_reward_array(self) -> np.ndarray:
        """Return the reward matrix for the wrapped environment.

        Returns
        -------
        np.ndarray
            The reward matrix of shape (n_states, n_actions).
        """
        raise NotImplementedError()
