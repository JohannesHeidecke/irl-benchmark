'''Relative Entropy IRL (Boularias et al., 2011).'''
import numpy as np
import gym
import time

from irl_benchmark.irl.algorithms.base_algorithm import BaseIRLAlgorithm
from irl_benchmark.irl.collect import collect_trajs
from irl_benchmark.irl.feature.feature_wrapper import FeatureWrapper
from irl_benchmark.irl.reward.reward_function import FeatureBasedRewardFunction
from irl_benchmark.rl.algorithms import RandomAgent
from irl_benchmark.utils.utils import unwrap_env
from irl_benchmark.utils.utils import sigma


class RelEnt(BaseIRLAlgorithm):
    """Relative Entropy IRL (Boularias et al., 2011).

    Assumes that reward is linear in features. Model-free,
    i.e. doesn't require transition dynamics.

    Based on minimizing the relative entropy (=KL divergence) between
    a baseline policy and a policy that matches the feature counts of
    the expert demonstrations. This minimization is done by
    subgradient descent, which is done approximately via importance
    sampling.
    """
    def __init__(self, env, expert_trajs, rl_alg_factory, gamma=.99,
                 horizon=20, delta=.05, eps=None):
        '''Set environment, RL agent factory, expert trajectories, and parameters.

        Args:
        env -- environment
        expert_trajs -- `list` of expert trajectories
        rl_alg_factory -- function that takes an environment
                          and returns an RL agent
        gamma -- `float`, discount factor
        horizon -- `int`, fixed length of trajectories to be considered
        delta -- confidence that feature count difference between output policy
                 and expert policy is less than 2 * epsilon
        calc_eps -- `float` or None; if None, then epsilons will be calculated
                    to guarantee matching expert feature counts within epsilon
                    with confidence delta (via Hoeffding's inequality). But
                    this requires the range of feature values.

        NOTE: Performance of the algorithm might depend on epsilons in a way
        I don't currently understand, as the epsilons occur in the expression
        used to approximate the relevant subgradient (ibid., p. 187, eq. 7).
        '''
        # Initialize base class and put remaining args into attributes.
        super(RelEnt, self).__init__(env, expert_trajs, rl_alg_factory)
        self.gamma = gamma
        self.horizon = horizon
        self.delta = delta

        # Compute remaining attributes.
        # Set gym.Env and FeatureWrapper envs as attributes.
        self.env_gym = unwrap_env(self.env, gym.Env)
        self.env_feat = unwrap_env(self.env, FeatureWrapper)
        # Set numbers of states, actions, expert trajs, and features.
        self.n_states = self.env_gym.observation_space.n
        self.n_actions = self.env_gym.action_space.n
        self.n_trajs = len(self.expert_trajs)
        self.n_features = self.env_feat.feature_shape()[0]
        assert isinstance(self.n_features, int)  # Should be dim of vector.
        # Calculate expert feature counts.
        self.expert_feature_count = self.feature_count(self.expert_trajs)
        # Set tolerance epsilon (one per feature) for not matching
        # expert feature counts.
        self.epsilons = np.zeros(self.n_features)
        if eps is not None:
            self.epsilons = eps
        else:
            # Calculate epsilons via Hoeffding (ibid., p. 184).
            max_features = self.env_feat.feature_range[1]
            min_features = self.env_feat.feature_range[0]
            self.epsilons = max_features - min_features
            scale = np.sqrt(-np.log(1 - self.delta) / (2 * self.n_trajs))
            scale *= (self.gamma ** (self.horizon+1) - 1) / (self.gamma - 1)
            self.epsilons *= scale

    def subgradients(self, trajs, reward_coefficients, policy):
        '''Return sample-based subgradients as in ibid., p. 185, eq. 8.

        Args:
        trajs -- `list` of trajectories
        reward_coefficients -- `np.ndarray` of shape (self.n_features, )
        policy -- policy from which trajs have been sampled, for now as table

        Returns:
        subgrads -- `np.ndarray` of shape (self.n_features, )
        '''
        # Get feature counts of input trajectories.  Note that we need
        # the counts of individual trajectories rather than averaging
        # over them. Also compute fraction U/pi from paper for each traj.
        feature_counts = np.zeros((len(trajs), self.n_features))
        fracs = np.zeros((1, len(trajs)))
        for i in range(len(trajs)):
            feature_counts[i] = self.feature_count([trajs[i]])
            U = self.joint_prob_of_actions(trajs[i])
            pi = self.policy_prob_of_actions(trajs[i], policy)
            fracs[i] = U / pi

        exp = np.exp(np.sum(reward_coefficients * feature_counts, axis=1))
        denom = fracs * exp
        numer = denom * feature_counts
        frac = np.sum(numer, axis=0) / np.sum(denom, axis=0)
        alphas = sigma(reward_coefficients)
        subgrads = self.expert_feature_count - frac - alphas * self.epsilons
        return subgrads

    def train(self, step_size=1e-4, time_limit=60, n_trajs=1000):
        '''Train for at most time_limit seconds w/ n_trajs non-expert trajs.

        Args:
        step_size -- `float`, size of each gradient descent step
        time_limit -- `int`, number of seconds to train
        n_trajs -- `int`, number of non-expert trajs to be collected

        Returns nothing.
        '''
        t0 = time.time()
        # Initialize normally distributed random reward coefficients.
        reward_coefficients = np.random.randn(self.n_features)

        # Get trajectories from random baseline policy.
        baseline_agent = RandomAgent(self.env)
        trajs = collect_trajs(self.env, baseline_agent, n_trajs, self.horizon)

        # Estimate subgradient based on collected trajectories, then
        # update reward coefficients.
        while time.time() < t0 + time_limit:
            subgrads = self.subgradients(trajs, reward_coefficients,
                                         baseline_agent.policy)
            reward_coefficients += step_size * subgrads

        self.reward_function = FeatureBasedRewardFunction(
            self.env, reward_coefficients)
        self.env.update_reward_function(self.reward_function)
