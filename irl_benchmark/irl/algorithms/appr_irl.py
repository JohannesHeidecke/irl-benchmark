"""Module for apprenticeship learning IRL."""
from typing import Callable, Dict, List, Tuple

import gym

from irl_benchmark.irl.algorithms.base_algorithm import BaseIRLAlgorithm, IRL_CONFIG_DOMAINS
from irl_benchmark.irl.reward.reward_function import BaseRewardFunction
from irl_benchmark.rl.algorithms import BaseRLAlgorithm


class ApprIRL(BaseIRLAlgorithm):
    """
    Apprenticeship learning IRL (Abbeel & Ng, 2004).

    Assumes reward linear in features.
    Two variants are implemented: SVM-based and projection-based.
    """

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
            A dictionary containing hyper-parameters for the algorithm.
        """
        super(ApprIRL, self).__init__(env, expert_trajs, rl_alg_factory,
                                      config)


    def train(self, no_irl_iterations: int,
              no_rl_episodes_per_irl_iteration: int,
              no_irl_episodes_per_irl_iteration: int
              ) -> Tuple[BaseRewardFunction, BaseRLAlgorithm]:
        pass


IRL_CONFIG_DOMAINS[ApprIRL] = {
    'gamma': {
        'type': float,
        'min': 0.0,
        'max': 1.0,
        'default': 0.9,
    },
    'mode': {
        'type': 'categorical',
        'values': ['svm', 'projection'],
        'default': 'svm',
    },
    'epsilon': {
        'type': float,
        'min': 0.0,
        'max': float('inf'),
        'default': 1e-6,
    },
}
