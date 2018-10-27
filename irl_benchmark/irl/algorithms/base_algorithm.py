"""Module for the abstract base class of all IRL algorithms."""
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Tuple, Union

import gym
import numpy as np

from irl_benchmark.config import preprocess_config, IRL_CONFIG_DOMAINS
from irl_benchmark.irl.feature.feature_wrapper import FeatureWrapper
from irl_benchmark.irl.reward.reward_function import BaseRewardFunction
from irl_benchmark.irl.reward.reward_wrapper import RewardWrapper
from irl_benchmark.rl.algorithms.base_algorithm import BaseRLAlgorithm
from irl_benchmark.utils.wrapper_utils import is_unwrappable_to, unwrap_env


class BaseIRLAlgorithm(ABC):
    """The abstract base class for all IRL algorithms."""

    def __init__(self, env: gym.Env, expert_trajs: List[Dict[str, list]],
                 rl_alg_factory: Callable[[gym.Env], BaseRLAlgorithm],
                 config: Union[dict, None] = None):
        """

        Parameters
        ----------
        env: gym.Env
            The gym environment to be trained on.
            Needs to be wrapped in a RewardWrapper to not leak the true reward function.
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
        self.config = preprocess_config(self, IRL_CONFIG_DOMAINS, config)

    @abstractmethod
    def train(self, no_irl_iterations: int,
              no_rl_episodes_per_irl_iteration: int,
              no_irl_episodes_per_irl_iteration: int
              ) -> Tuple[BaseRewardFunction, BaseRLAlgorithm]:
        """Train the IRL algorithm.

        Parameters
        ----------
        no_irl_iterations: int
            The number of iteration the algorithm should be run.
        no_rl_episodes_per_irl_iteration: int
            The number of episodes the RL algorithm is allowed to run in
            each iteration of the IRL algorithm.
        no_irl_episodes_per_irl_iteration: int
            The number of episodes permitted to be run in each iteration
            to update the current reward estimate (e.g. to estimate state frequencies
            of the currently optimal policy).

        Returns
        -------
        Tuple[BaseRewardFunction, BaseRLAlgorithm]
            The estimated reward function and a RL agent trained for this estimate.
        """
        raise NotImplementedError()

    def feature_count(self, trajs: List[Dict[str, list]],
                      gamma: float) -> np.ndarray:
        """Return empirical discounted feature counts of input trajectories.

        Parameters
        ----------
        trajs: List[Dict[str, list]]
             A list of trajectories.
            Each trajectory is a dictionary with keys
            ['states', 'actions', 'rewards', 'true_rewards', 'features'].
            The values of each dictionary are lists.
            See :func:`irl_benchmark.irl.collect.collect_trajs`.
        gamma: float
            The discount factor. Must be in range [0., 1.].

        Returns
        -------
        np.ndarray
            A numpy array containing discounted feature counts. The shape
            is the same as the trajectories' feature shapes. One scalar
            feature count per feature.
        """
        # Initialize feature count sum to zeros of correct shape:
        feature_count_sum = np.zeros(
            unwrap_env(self.env, FeatureWrapper).feature_shape())

        for traj in trajs:

            assert traj['features']  # empty lists are False in python

            # gammas is a vector containing [gamma^0, gamma^1, gamma^2, ... gamma^l]
            # where l is length of the trajectory:
            gammas = gamma**np.arange(len(traj['features']))
            traj_feature_count = np.sum(
                gammas.reshape(-1, 1) * np.array(traj['features']), axis=0)
            # add trajectory's feature count:
            feature_count_sum += traj_feature_count
        # divide feature_count_sum by number of trajectories to normalize:
        feature_count = feature_count_sum / len(trajs)
        return feature_count
