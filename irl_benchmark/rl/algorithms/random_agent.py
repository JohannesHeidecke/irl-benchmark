"""Module for a randomly acting RL agent."""
from typing import Union

import gym
import numpy as np

from irl_benchmark.config import RL_CONFIG_DOMAINS
from irl_benchmark.rl.algorithms.base_algorithm import BaseRLAlgorithm


class RandomAgent(BaseRLAlgorithm):
    """Random agent. Picks actions uniformly random.
    """

    def __init__(self, env: gym.Env, config: dict = {}):
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
        # TODO: do some environments return lists when sampling an action? adapt signature also in base alg
        return self.env.action_space.sample()

    def policy(self, state: int) -> np.ndarray:
        raise NotImplementedError()


RL_CONFIG_DOMAINS[RandomAgent] = {}
