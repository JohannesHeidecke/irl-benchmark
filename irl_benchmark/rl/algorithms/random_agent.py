"""Module for a randomly acting RL agent."""
from typing import Union

import gym
from gym.spaces import Box, Discrete, MultiDiscrete, MultiBinary
import numpy as np

from irl_benchmark.config import RL_CONFIG_DOMAINS
from irl_benchmark.rl.algorithms.base_algorithm import BaseRLAlgorithm


class RandomAgent(BaseRLAlgorithm):
    """Random agent. Picks actions uniformly random.
    """

    def __init__(self, env: gym.Env, config: dict = None):
        """

        Parameters
        ----------
        env: gym.Env
            A gym environment
        config: dict
            Configuration of hyperparameters.
            Can be empty dictionary, random agent doesn't have hyperparams.
        """
        super(RandomAgent, self).__init__(env, config)
        assert hasattr(self.env.action_space, 'sample')

    def train(self, no_episodes: int):
        """ Random agent does not need to be trained."""
        pass

    def pick_action(self, state: Union[int, float, np.ndarray]
                    ) -> Union[int, float, np.ndarray]:
        """ Pick an action given a state.

        Picks uniformly random from all possible actions, using the environments
        action_space.sample() method.

        Parameters
        ----------
        state: int
            An integer corresponding to a state of a DiscreteEnv.
            Not used in this agent.

        Returns
        -------
        Union[int, float, np.ndarray]
            An action
        """
        # if other spaces are needed, check if their sample method conforms with
        # returned type, change if necessary.
        assert isinstance(self.env.action_space,
                          (Box, Discrete, MultiDiscrete, MultiBinary))
        return self.env.action_space.sample()

    def policy(self, state: int) -> np.ndarray:
        raise NotImplementedError()


RL_CONFIG_DOMAINS[RandomAgent] = {}
