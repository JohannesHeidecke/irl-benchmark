import gym
import numpy as np
import time

from irl_benchmark.irl.algorithms.base_algorithm import BaseIRLAlgorithm
from irl_benchmark.irl.reward.reward_function import FeatureBasedRewardFunction
from irl_benchmark.utils.utils import unwrap_env

class MaxEnt(BaseIRLAlgorithm):
    '''Maximum Entropy IRL (Ziebart et al., 2008).

    Not to be confused with Maximum Entropy Deep IRL (Wulfmeier et al., 2016)
    or Maximum Causal Entropy IRL (Ziebart et al., 2010).
    '''

    def __init__(self,
            env,
            expert_trajs,
            transition_dynamics,
            rl_alg_factory,
            gamma=0.99,
            lr=0.02
        ):
        '''Initialize Maximum Entropy IRL algorithm. '''
        super(MaxEnt, self).__init__(env, expert_trajs, rl_alg_factory)
        # make sure env is DiscreteEnv (other cases not implemented yet
        # TODO: implement other cases
        unwrap_env(env, gym.envs.toy_text.discrete.DiscreteEnv)

        self.gamma = gamma
        self.transition_dynamics = transition_dynamics
        self.n_states, self.n_actions, _ = np.shape(self.transition_dynamics)
        self.lr = lr

    def expected_svf(self, policy):
        '''Calculate the expected state visitation frequency for the trajectories
        under the given policy.

        Returns vector of state visitation frequencies.
        ''' 

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
                        tot += mu[pre_s, t - 1] * self.transition_dynamics[pre_s, action, s] * policy[pre_s, action]
                mu[s, t] = tot
        return np.sum(mu, 1)

    def train(self, feat_map, time_limit=300, rl_time_per_iteration=15, verbose=False):
        """
        Maximum Entropy Inverse Reinforcement Learning (Maxent IRL)
        inputs:
          feat_map    NxD matrix - the features for each state
          P_a         NxN_ACTIONSxN matrix - P_a[s0, a, s1] is the transition prob of
                                             landing at state s1 when taking action
                                             a at state s0
          gamma       float - RL discount factor
          trajs       a list of demonstrations
          lr          float - learning rate
          n_iters     int - number of optimization steps
        returns
          rewards     Nx1 vector - recovered state rewards
        """
        t0 = time.time()

        # init parameters
        theta = np.random.uniform(size=(feat_map.shape[1],))
        

        # calc feature expectations
        feat_exp = np.zeros([feat_map.shape[1]])
        for episode in self.expert_trajs:
            for state in episode['states']:
                feat_exp += feat_map[state]
        feat_exp = feat_exp / len(self.expert_trajs)
        #print(feat_exp)

        agent = self.rl_alg_factory(self.env)

        # training
        iteration_counter = 0
        while time.time() < t0 + time_limit:
            iteration_counter += 1
            if verbose:
                print('iteration: {}'.format(iteration_counter))

            reward_function_estimate = FeatureBasedRewardFunction(self.env, theta)
            self.env.update_reward_function(reward_function_estimate)

            # compute policy
            agent.train(time_limit=rl_time_per_iteration)

            policy = agent.pi

            # compute state visitation frequencies
            svf = self.expected_svf(policy)

            # compute gradients
            grad = -(feat_exp - feat_map.T.dot(svf))

            # update params
            theta += self.lr * grad

        # return sigmoid(normalize(rewards))
        self.reward_function = reward_function_estimate
        return theta
