"""Module for Maximum Causal Entropy IRL."""
from typing import Callable, Dict, List

import gym
import numpy as np

from irl_benchmark.config import IRL_CONFIG_DOMAINS, IRL_ALG_REQUIREMENTS
from irl_benchmark.irl.algorithms.base_algorithm import BaseIRLAlgorithm
from irl_benchmark.irl.feature.feature_wrapper import FeatureWrapper
from irl_benchmark.irl.reward.reward_function import FeatureBasedRewardFunction
from irl_benchmark.irl.reward.reward_wrapper import RewardWrapper
from irl_benchmark.metrics.base_metric import BaseMetric
from irl_benchmark.rl.algorithms.base_algorithm import BaseRLAlgorithm
from irl_benchmark.rl.model.model_wrapper import BaseWorldModelWrapper
from irl_benchmark.utils.wrapper import unwrap_env


class MaxCausalEntIRL(BaseIRLAlgorithm):
    """Maximum Entropy IRL (Ziebart et al., 2008).

    Not to be confused with Maximum Entropy Deep IRL (Wulfmeier et al., 2016)
    or Maximum Causal Entropy IRL (Ziebart et al., 2010).
    """

    def __init__(self, env: gym.Env, expert_trajs: List[Dict[str, list]],
                 rl_alg_factory: Callable[[gym.Env], BaseRLAlgorithm],
                 metrics: List[BaseMetric], config: dict):

        super(MaxCausalEntIRL, self).__init__(env, expert_trajs,
                                              rl_alg_factory, metrics, config)

        # get transition matrix (with absorbing state)
        self.transition_matrix = unwrap_env(
            env, BaseWorldModelWrapper).get_transition_array()
        self.n_states, self.n_actions, _ = self.transition_matrix.shape

        # get map of features for all states:
        feature_wrapper = unwrap_env(env, FeatureWrapper)
        self.feat_map = feature_wrapper.feature_array()

    def sa_visitations(self):
        """
        Given a list of trajectories in an MDP, computes the state-action
        visitation counts and the probability of a trajectory starting in state s.

            Arrays of shape (n_states, n_actions) and (n_states).
        """

        s0_count = np.zeros(self.n_states)
        sa_visit_count = np.zeros((self.n_states, self.n_actions))

        for traj in self.expert_trajs:
            # traj['states'][0] is the state of the first timestep of the trajectory.
            s0_count[traj['states'][0]] += 1

            for timestep in range(len(traj['actions'])):
                state = traj['states'][timestep]
                action = traj['actions'][timestep]

                sa_visit_count[state, action] += 1

        # Count into probability
        P0 = s0_count / len(self.expert_trajs)

        return sa_visit_count, P0

    def occupancy_measure(self,
                          policy,
                          initial_state_dist,
                          t_max=None,
                          threshold=1e-6):
        """
        Computes occupancy measure of a MDP under a given time-constrained policy
        -- the expected discounted number of times that policy π visits state s in
        a given number of timesteps, as in Algorithm 9.3 of Ziebart's thesis:
        http://www.cs.cmu.edu/~bziebart/publications/thesis-bziebart.pdf.
        """

        if initial_state_dist is None:
            initial_state_dist = np.ones(self.n_states) / self.n_states
        d_prev = np.zeros_like(initial_state_dist)

        t = 0

        diff = float("inf")

        while diff > threshold:

            d = np.copy(initial_state_dist)

            for state in range(self.n_states):
                for action in range(self.n_actions):
                    # for all next_state reachable:

                    for next_state in range(self.n_states):
                        # probabilty of reaching next_state by taking action
                        prob = self.transition_matrix[state, action,
                                                      next_state]
                        d[next_state] += self.config['gamma'] * d_prev[
                            state] * policy[state, action] * prob

            diff = np.amax(abs(d_prev - d))  # maxima of the flattened array
            d_prev = np.copy(d)

            if t_max is not None:
                t += 1
                if t == t_max:
                    break
        return d_prev

    def train(self, no_irl_iterations: int,
              no_rl_episodes_per_irl_iteration: int,
              no_irl_episodes_per_irl_iteration: int):
        """

        """

        sa_visit_count, P0 = self.sa_visitations()

        # calculate feature expectations
        expert_feature_count = self.feature_count(self.expert_trajs, gamma=1.0)

        # initialize the parameters
        reward_function = FeatureBasedRewardFunction(self.env, 'random')
        theta = reward_function.parameters

        agent = self.rl_alg_factory(self.env)

        irl_iteration_counter = 0

        while irl_iteration_counter < no_irl_iterations:
            irl_iteration_counter += 1

            if self.config['verbose']:
                print('IRL ITERATION ' + str(irl_iteration_counter))

            reward_wrapper = unwrap_env(self.env, RewardWrapper)
            reward_wrapper.update_reward_parameters(theta)

            # compute policy
            agent.train(no_rl_episodes_per_irl_iteration)

            policy = agent.policy_array()
            state_values = agent.state_values
            q_values = agent.q_values

            # occupancy measure
            d = self.occupancy_measure(
                policy=policy, initial_state_dist=P0)[:-1]

            # log-likeilihood gradient
            grad = -(expert_feature_count - np.dot(self.feat_map.T, d))

            # graduate descent
            theta -= self.config['lr'] * grad

            evaluation_input = {
                'irl_agent': agent,
                'irl_reward': reward_function
            }
            self.evaluate_metrics(evaluation_input)

        return theta


IRL_CONFIG_DOMAINS[MaxCausalEntIRL] = {
    'gamma': {
        'type': float,
        'min': 0.0,
        'max': 1.0,
        'default': 0.9,
    },
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

IRL_ALG_REQUIREMENTS[MaxCausalEntIRL] = {
    'requires_features': True,
    'requires_transitions': True,
}
