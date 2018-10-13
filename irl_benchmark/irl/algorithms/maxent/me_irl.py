import gym
import numpy as np
import time

from irl_benchmark.irl.algorithms.base_algorithm import BaseIRLAlgorithm
from irl_benchmark.irl.reward.reward_function import FeatureBasedRewardFunction
from irl_benchmark.utils.utils import unwrap_env, is_unwrappable_to


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
                 learning_rate=0.02):
        '''Initialize Maximum Entropy IRL algorithm.

        Args:
        env -- `RewardWrapper` around a DiscreteEnv from OpenAI Gym
        expert_trajs -- `list` of expert trajectories as dictionaries
        transition_dynamics -- `np.ndarray` of shape [n_states, n_actions,
                               n_states],
                               matrix of transition probabilities in env
        rl_alg_factory -- `function` taking an env and returning an RL agent
        learning_rate -- `float`, learning rate for gradient descent
        '''
        super(MaxEnt, self).__init__(env, expert_trajs, rl_alg_factory)
        # make sure env is DiscreteEnv (other cases not implemented yet)
        # TODO: implement other cases
        is_unwrappable_to(env, gym.envs.toy_text.discrete.DiscreteEnv)

        self.transition_dynamics = transition_dynamics
        self.n_states = unwrap_env(env, gym.Env).observation_space.n
        self.n_actions = unwrap_env(env, gym.Env).action_space.n
        self.learning_rate = learning_rate

    def expected_svf(self, policy):
        '''Calculate the expected state visitation frequency for the trajectories
        under the given policy.

        Args:
        policy -- `np.ndarray` of shape [self.n_states, self.n_actions],
                  entry [s, a] is prob of taking action a in state s

        Returns:
        mu -- `np.ndarray` of shape (self.n_states, ),
              vector of approximated state visitation frequencies
        '''
        # get the length of longest trajectory:
        longest_traj_len = 1  # init
        for traj in self.expert_trajs:
            longest_traj_len = max(longest_traj_len, len(traj['states']))
        longest_traj_len *= 100  # To catch policies that take longer.

        # mu[s, t] is the prob of visiting state s at time t
        mu = np.zeros((self.n_states, longest_traj_len))

        # Get empirical initial state distribution.
        for traj in self.expert_trajs:
            mu[traj['states'][0], 0] += 1
        mu[:, 0] = mu[:, 0] / len(self.expert_trajs)

        # Calculate the state-visit freqs based on policy & dynamics.
        #
        # Note that this isn't exact because we ignore trajectories that
        # have positive probability but are longer than any expert trajectory.
        #
        # The approximation should be good when the input policy is
        # close to the expert one and we use a large number of expert
        # trajectories, but can be bad otherwise.
        #
        # We ad hoc mitigated this issue above by considering trajectories
        # up to 10 times the length of the longest expert trajectory.
        for t in range(1, longest_traj_len):
            for s in range(self.n_states):
                tot = 0
                for pre_s in range(self.n_states):
                    for action in range(self.n_actions):
                        tot += mu[pre_s, t - 1] * self.transition_dynamics[
                            pre_s, action, s] * policy[pre_s, action]
                mu[s, t] = tot
        mu = np.sum(mu, axis=1)
        return mu

    def train(self,
              feat_map,
              time_limit=300,
              rl_time_per_iteration=15,
              verbose=False):
        """Maximum Entropy Inverse Reinforcement Learning (MaxEnt IRL).

        Args:
        feat_map -- `np.ndarray` of shape [self.n_states, k],
                    row s contains k-dimensional features for state s
        time_limit -- `int`, total IRL training time in seconds
        rl_time_per_iteration -- `int`, per-iteration training time of the
                                 Reinforcement Learning step
        verbose -- `bool`, print stuff if true

        Returns:
        rewards -- `np.ndarray` of shape (self.n_states, ),
                   recovered state rewards
        """
        t0 = time.time()

        # init parameters
        theta = np.random.uniform(size=(feat_map.shape[1], ))

        # calc undiscounted feature expectations
        feat_exp = np.zeros([feat_map.shape[1]])
        for episode in self.expert_trajs:
            for state in episode['states']:
                feat_exp += feat_map[state]
        feat_exp = feat_exp / len(self.expert_trajs)
        if verbose:
            print('Undiscounted expert feature expectations:')
            print(feat_exp)

        agent = self.rl_alg_factory(self.env)

        # training
        iteration_counter = 0
        while time.time() < t0 + time_limit:
            iteration_counter += 1
            if verbose and iteration_counter % 20 == 0:
                print('iteration: {}'.format(iteration_counter))

            reward_function_estimate = FeatureBasedRewardFunction(
                self.env, theta)
            self.env.update_reward_function(reward_function_estimate)

            # compute policy
            agent.train(time_limit=rl_time_per_iteration)

            # Note that this currently only works for value_iteration.
            policy = agent.pi

            # compute state visitation frequencies
            svf = self.expected_svf(policy)

            # compute gradients
            grad = feat_exp - feat_map.T.dot(svf)

            # update params
            theta += self.learning_rate * grad

        return theta
