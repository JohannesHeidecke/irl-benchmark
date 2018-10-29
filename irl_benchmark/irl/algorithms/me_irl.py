import gym
import numpy as np
from typing import Callable, Dict, List, Tuple

from irl_benchmark.config import IRL_CONFIG_DOMAINS
from irl_benchmark.irl.algorithms.base_algorithm import BaseIRLAlgorithm
from irl_benchmark.irl.reward.reward_function import FeatureBasedRewardFunction
from irl_benchmark.rl.algorithms.base_algorithm import BaseRLAlgorithm
from irl_benchmark.utils.wrapper_utils import get_transition_matrix


class MaxEntIRL(BaseIRLAlgorithm):

    def __init__(self, env: gym.Env, expert_trajs: List[Dict[str, list]],
                 rl_alg_factory: Callable[[gym.Env], BaseRLAlgorithm],
                 config: dict):

        super(MaxEntIRL, self).__init__(env, expert_trajs, rl_alg_factory,
                                      config)
        self.transition_matrix = get_transition_matrix(self.env)
        self.n_states, self.n_actions, _ = self.transition_matrix.shape
        self.feat_map = np.eye(self.n_states**2)

    def expected_svf(self, policy):
        """Calculate the expected state visitation frequency for the trajectories
        under the given policy.
        Returns vector of state visitation frequencies.
        """

        # get the length of longest trajectory:
        longest_traj_len = 1  # init
        for traj in self.expert_trajs:
            longest_traj_len = max(longest_traj_len, len(traj['states']))

        # mu[s, t] is the prob of visiting state s at time t
        mu = np.zeros((self.n_states, longest_traj_len))

        for traj in self.expert_trajs:
            mu[traj['states'][0], 0] += 1
        mu[:, 0] = mu[:, 0] / len(self.expert_trajs)

        for t in range(1, longest_traj_len):
            for s in range(self.n_states):
                tot = 0
                for pre_s in range(self.n_states):
                    for action in range(self.n_actions):
                        tot += mu[pre_s, t - 1] * self.transition_matrix[pre_s, action, s] * policy[pre_s, action]
                mu[s, t] = tot
        return np.sum(mu, 1)

    def train(self, no_irl_iterations: int,
              no_rl_episodes_per_irl_iteration: int,
              no_irl_episodes_per_irl_iteration: int
              ):

        # calc feature expectations
        feat_exp = np.zeros([self.feat_map.shape[1]])
        for episode in self.expert_trajs:
            for state in episode['states']:
                feat_exp += self.feat_map[state]
        feat_exp = feat_exp / len(self.expert_trajs)

        # start with an agent
        agent = self.rl_alg_factory(self.env)

        # init parameters
        theta = np.random.uniform(size=(self.feat_map.shape[1],))

        irl_iteration_counter = 0
        while irl_iteration_counter < no_irl_iterations:
            irl_iteration_counter += 1

            if self.config['verbose']:
                print('IRL ITERATION ' + str(irl_iteration_counter))

            reward_function_estimate = FeatureBasedRewardFunction(self.env, theta)
            self.env.update_reward_function(reward_function_estimate)

            # compute policy
            agent.train(no_rl_episodes_per_irl_iteration)

            policy = agent.policy_array()

            # compute state visitation frequencies
            svf = self.expected_svf(policy)

            # compute gradients
            grad = -(feat_exp - self.feat_map.T.dot(svf))

            # update params
            theta += self.config['lr'] * grad

        return theta


IRL_CONFIG_DOMAINS[MaxEntIRL] = {
    'gamma': {
        'type': float,
        'min': 0.0,
        'max': 1.0,
        'default': 0.9,
    },

    'epsilon': {
        'type': float,
        'min': 0.0,
        'max': float('inf'),
        'default': 1e-6,
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
