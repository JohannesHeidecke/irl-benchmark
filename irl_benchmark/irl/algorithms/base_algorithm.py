"""Module for the abstract base class of all IRL algorithms."""
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Tuple

import gym

from irl_benchmark.irl.reward.reward_function import BaseRewardFunction
from irl_benchmark.irl.reward.reward_wrapper import RewardWrapper
from irl_benchmark.rl.algorithms.base_algorithm import BaseRLAlgorithm
from irl_benchmark.utils import is_unwrappable_to


class BaseIRLAlgorithm(ABC):
    """The abstract base class for all IRL algorithms."""

    def __init__(self, env: gym.Env, expert_trajs: List[Dict[str, list]],
                 rl_alg_factory: Callable[[gym.Env], BaseRLAlgorithm],
                 config: dict):
        """

        Parameters
        ----------
        env: gym.Env
            The gym environment to be trained on.
            Needs to be wrapped in a RewardWrapper to prevent leaking the true reward function.
        expert_trajs: List[dict]
            A list of trajectories.
            Each trajectory is a dictionary with keys
            ['states', 'actions', 'rewards', 'true_rewards', 'features'].
            The values of each dictionary are lists.
            See :func:`irl_benchmark.irl.collect.collect_trajs`.
        rl_alg_factory: Callable[[gym.Env], BaseRLAlgorithm]
            A function which returns a new RL algorithm when called.
        config: dict
            A dictionary containing algorithm-specific parameters.
        """

        assert is_unwrappable_to(env, RewardWrapper)
        self.env = env
        self.expert_trajs = expert_trajs
        self.rl_alg_factory = rl_alg_factory
        self.config = config

    @abstractmethod
    def train(self, no_irl_iterations: int,
              no_rl_episodes_per_irl_iteration: int
              ) -> Tuple[BaseRewardFunction, BaseRLAlgorithm]:
        """Train the IRL algorithm.

        Parameters
        ----------
        no_irl_iterations: int
            The number of iteration the algorithm should be run.
        no_rl_episodes_per_irl_iteration: int
            The number of episodes the RL algorithm is allowed to run in
            each iteration of the IRL algorithm.

        Returns
        -------
        Tuple[BaseRewardFunction, BaseRLAlgorithm]
            The estimated reward function and a RL agent trained for this estimate.
        """
        raise NotImplementedError()
