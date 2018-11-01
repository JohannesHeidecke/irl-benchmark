"""Module for maximum entropy inverse reinforcement learning."""

from typing import Callable, Dict, List

import gym
from gym.envs.toy_text.discrete import DiscreteEnv
import numpy as np

from irl_benchmark.config import IRL_CONFIG_DOMAINS
from irl_benchmark.irl.algorithms.base_algorithm import BaseIRLAlgorithm
from irl_benchmark.irl.feature.feature_wrapper import FeatureWrapper
from irl_benchmark.irl.reward.reward_function import FeatureBasedRewardFunction
from irl_benchmark.metrics.base_metric import BaseMetric
from irl_benchmark.rl.algorithms.base_algorithm import BaseRLAlgorithm
from irl_benchmark.utils.wrapper import get_transition_matrix, is_unwrappable_to, unwrap_env


class MaxEntIRL(BaseIRLAlgorithm):
    """Maximum Entropy IRL (Ziebart et al., 2008).

    Not to be confused with Maximum Entropy Deep IRL (Wulfmeier et al., 2016)
    or Maximum Causal Entropy IRL (Ziebart et al., 2010).
    """

    def __init__(self, env: gym.Env, expert_trajs: List[Dict[str, list]],
                 rl_alg_factory: Callable[[gym.Env], BaseRLAlgorithm],
                 metrics: List[BaseMetric], config: dict):
        """See :class:`irl_benchmark.irl.algorithms.base_algorithm.BaseIRLAlgorithm`."""

        assert is_unwrappable_to(env, DiscreteEnv)
        assert is_unwrappable_to(env, FeatureWrapper)

        super(MaxEntIRL, self).__init__(env, expert_trajs, rl_alg_factory,
                                        metrics, config)
        # get transition matrix (with absorbing state)
        self.transition_matrix = get_transition_matrix(self.env)
        self.n_states, self.n_actions, _ = self.transition_matrix.shape

        # get map of features for all states:
        feature_wrapper = unwrap_env(env, FeatureWrapper)
        self.feat_map = feature_wrapper.feature_array()

    def expected_svf(self, policy: np.ndarray) -> np.ndarray:
        """Calculate the expected state visitation frequency for the trajectories
        under the given policy. Returns vector of state visitation frequencies.
        Uses self.transition_matrix.

        Parameters
        ----------
        policy: np.ndarray
            The policy for which to calculate the expected SVF.

        Returns
        -------
        np.ndarray
            Expected state visitation frequencies as a numpy array of shape (n_states,).
        """
        # get the length of longest trajectory:
        longest_traj_len = 1  # init
        for traj in self.expert_trajs:
            longest_traj_len = max(longest_traj_len, len(traj['states']))

        # svf[state, time] is the frequency of visiting a state at some point of time
        svf = np.zeros((self.n_states, longest_traj_len))

        for traj in self.expert_trajs:
            svf[traj['states'][0], 0] += 1
        svf[:, 0] = svf[:, 0] / len(self.expert_trajs)

        for time in range(1, longest_traj_len):
            for state in range(self.n_states):
                total = 0
                for previous_state in range(self.n_states):
                    for action in range(self.n_actions):
                        total += svf[
                            previous_state, time - 1] * self.transition_matrix[
                                previous_state, action, state] * policy[
                                    previous_state, action]
                svf[state, time] = total
        # sum over all time steps and return SVF for each state:
        return np.sum(svf, axis=1)

    def train(self, no_irl_iterations: int,
              no_rl_episodes_per_irl_iteration: int,
              no_irl_episodes_per_irl_iteration: int):
        """Train algorithm. See abstract base class for parameter types."""

        # calculate feature expectations
        expert_feature_count = self.feature_count(self.expert_trajs, gamma=1.0)

        # start with an agent
        agent = self.rl_alg_factory(self.env)

        reward_function = FeatureBasedRewardFunction(self.env, 'random')
        self.env.update_reward_function(reward_function)
        theta = reward_function.parameters

        irl_iteration_counter = 0
        while irl_iteration_counter < no_irl_iterations:
            irl_iteration_counter += 1

            if self.config['verbose']:
                print('IRL ITERATION ' + str(irl_iteration_counter))
            # compute policy
            agent.train(no_rl_episodes_per_irl_iteration)

            policy = agent.policy_array()

            # compute state visitation frequencies, discard absorbing state
            svf = self.expected_svf(policy)[:-1]

            # compute gradients
            grad = (expert_feature_count - self.feat_map.T.dot(svf))

            # update params
            theta += self.config['lr'] * grad

            reward_function.update_parameters(theta)

            evaluation_input = {
                'irl_agent': agent,
                'irl_reward': reward_function
            }
            self.evaluate_metrics(evaluation_input)

        return theta


IRL_CONFIG_DOMAINS[MaxEntIRL] = {
    'verbose': {
        'type': bool,
        'default': True
    },
    'lr': {
        'type': float,
        'default': 0.02,
        'min': 0.000001,
        'max': 50
    }
}
