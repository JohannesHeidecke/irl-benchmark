"""A module containing base classes for RL algorithms"""

from abc import ABC, abstractmethod
from typing import Union

import gym
import numpy as np

from irl_benchmark.config import preprocess_config, RL_CONFIG_DOMAINS


class BaseRLAlgorithm(ABC):
    """Base class for Reinforcement Learning agents."""

    def __init__(self, env: gym.Env, config: Union[dict, None] = None):
        """

        Parameters
        ----------
        env: gym.Env
            The gym environment the agent will be trained on.
        config: dict
            A dictionary of algorithm-specific parameters.
        """
        self.env = env
        self.config = preprocess_config(self, RL_CONFIG_DOMAINS, config)

    @abstractmethod
    def train(self, no_episodes: int):
        """ Train the agent.

        Parameters
        ----------
        no_episodes: int
            Training will be run for this number of episodes.
        """
        raise NotImplementedError()

    @abstractmethod
    def pick_action(self, state: Union[int, float, np.ndarray]
                    ) -> Union[int, float, np.ndarray]:
        """ Pick an action given a state.

        Parameters
        ----------
        state: Union[int, float, np.ndarray]
            A state of the environment, compatible with env.observation_space.

        Returns
        -------
        Union[int, float, np.ndarray]
            An action, compatible with env.action_space

        """
        raise NotImplementedError()

    @abstractmethod
    def policy(self, state: Union[int, float, np.ndarray]) -> np.ndarray:
        """Return the probabilities of picking actions given a state.

        NOTE: This is currently only defined for discrete action spaces.

        Parameters
        ----------
        state: Union[int, float, np.ndarray]
            A state of the environment, compatible with env.observation_space.

        Returns
        -------
        np.ndarray
            A numpy ndarray containing probabilities of each action.
            The shape of the array corresponds to the dimension of env.action_space.

        """
        raise NotImplementedError()
