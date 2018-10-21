"""Module for the abstract base class of all IRL algorithms."""
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Tuple

import gym

from irl_benchmark.irl.reward.reward_function import BaseRewardFunction
from irl_benchmark.irl.reward.reward_wrapper import RewardWrapper
from irl_benchmark.rl.algorithms.base_algorithm import BaseRLAlgorithm
from irl_benchmark.utils import is_unwrappable_to

# A dictionary containing allowed and default values for
# the config of each IRL algorithm.
# To be extended in each specific IRL algorithm implementation:
IRL_CONFIG_DOMAINS = {}


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
        self.config = self.preprocess_config(config)

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

    def preprocess_config(self, config: dict) -> dict:
        """ Pre-processes a config dictionary.

        This is based on the values specified for each IRL algorithm in
        IRL_CONFIG_DOMAINS. The following steps are performed:

        * If values in config are missing, add their default values.
        * If values are illegal (e.g. too high), raise an error.
        * If unknown fields are specified, raise an error.

        Manipulates the passed config in-place (deep-copy it if undesired).

        Parameters
        ----------
        config: dict
            The unprocessed config dictionary.

        Returns
        -------
        dict
            The processed config dictionary.

        """
        # get config domain for the correct subclass calling this:
        config_domain: dict = IRL_CONFIG_DOMAINS[type(self)]
        for key in config_domain.keys():
            if key in config.keys():
                # for numerical fields:
                if config_domain[key]['type'] in [float, int]:
                    # check if right type:
                    assert type(config[key]) is config_domain[key][
                        'type'], "Wrong config value type for key " + str(key)
                    # check if value is high enough:
                    assert config[key] >= config_domain[key][
                        'min'], "Config value too low for key " + str(key)
                    # check if value is low enough:
                    assert config[key] <= config_domain[key][
                        'max'], "Config value too high for key " + str(key)
                # for categorical fields:
                elif config_domain[key]['type'] == 'categorical':
                    # check if value is allowed:
                    assert config[key] in config_domain[key][
                        'values'], "Illegal config value : " + config[key]
                else:
                    # encountered type for which no implementation has been written
                    # extend code above to fix.
                    raise NotImplementedError(
                        "No implementation for config value type: " +
                        str(config_domain[key]['type']))
            else:
                # key not specified in given config, use default value:
                config[key] = config_domain[key]['default']
        # check if config only contains legal fields:
        for key in config.keys():
            if key not in config_domain.keys():
                raise ValueError("Unknown config field: " + str(key))

        # return the pre-processed config:
        return config
