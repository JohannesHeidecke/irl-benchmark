import numpy as np

from irl_benchmark.irl.algorithms.base_algorithm import BaseIRLAlgorithm
from irl_benchmark.irl.reward.reward_function import FeatureBasedRewardFunction
from irl_benchmark.rl.algorithms.value_iteration import ValueIteration

import numpy as np


class MaxEnt(BaseIRLAlgorithm):
    """Maximum Entropy IRL (Ziebart et al., 2008).

    Not to be confused with Maximum Entropy Deep IRL (Wulfmeier et al., 2016)
    or Maximum Causal Entropy IRL (Ziebart et al., 2010).
    """

    def __init__(self, env, transition_dynamics, expert_trajs, rl_alg_factory, gamma=.99, lr=0.02):
        super(MaxEnt, self).__init__(env, expert_trajs, rl_alg_factory= rl_alg_factory)
        self.gamma = gamma
        self.transition_dynamics = transition_dynamics
        self.n_states, self.n_actions, _ = np.shape(self.transition_dynamics)
        self.lr = lr

    def expected_svf(self, policy):
        """
        calculates the expected state visitation frequency for the trajectories
        under the given policy
        :param policy:
        :return:
        p   Nx1 vector - state visitation frequencies
        """

        # get the length of longest trajectory
        longest_traj_len = 1  # init
        for traj in self.expert_trajs:
            longest_traj_len = max(longest_traj_len, len(traj['states']))

        n_t = longest_traj_len

        # mu[s, t] is the prob of visiting state s at time t
        mu = np.zeros((self.n_states, n_t))

        for traj in self.expert_trajs:
            for state in traj['states']:
                mu[state, 0] += 1
        mu[:, 0] = mu[:, 0] / len(self.expert_trajs)

        for t in range(1, n_t):
            for s in range(self.n_states):
                tot = 0
                for pre_s in range(self.n_states):
                    for action in range(self.n_actions):
                        tot += mu[pre_s, t - 1] * self.transition_dynamics[pre_s, action, s] * policy[pre_s, action]
                mu[s, t] = tot
        return np.sum(mu, 1)

    def train(self, feat_map, time_limit=300, n_iters=30):

        """
        Maximum Entropy Inverse Reinforcement Learning (Maxent IRL)
        inputs:
          feat_map    NxD matrix - the features for each state

          gamma       float - RL discount factor
          trajs       a list of demonstrations
          lr          float - learning rate
          n_iters     int - number of optimization steps
        returns
          rewards     Nx1 vector - recovered state rewards
        """
        # init parameters
        theta = np.random.uniform(size=(feat_map.shape[1],))
        reward_function_estimate = FeatureBasedRewardFunction(self.env, theta)

        self.env.update_reward_function(reward_function_estimate)

        # calc feature expectations
        feat_exp = np.zeros([feat_map.shape[1]])
        for episode in self.expert_trajs:
            for state in episode['states']:
                feat_exp += feat_map[state]
        feat_exp = feat_exp / len(self.expert_trajs)
        #print(feat_exp)

        agent = self.rl_alg_factory(self.env)

        # training
        for iteration in range(n_iters):

            if iteration % (n_iters / 20) == 0:
                print('iteration: {}/{}'.format(iteration, n_iters))

            # compute reward function
            rewards = np.dot(feat_map, theta)


            # compute policy
            agent.train(time_limit= 200)

            policy = agent.pi

            # compute state visitation frequencies
            svf = self.expected_svf(policy)

            # compute gradients
            grad = -(feat_exp - feat_map.T.dot(svf))

            # update params
            theta += self.lr * grad

            #rewards = np.dot(feat_map, theta)
        # return sigmoid(normalize(rewards))
        return rewards
