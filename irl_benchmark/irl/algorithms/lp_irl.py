import numpy as np
import gym
from typing import Callable, Dict, List, Tuple
from irl_benchmark.irl.algorithms.base_algorithm import BaseIRLAlgorithm
from irl_benchmark.metrics.base_metric import BaseMetric
from irl_benchmark.irl.reward.reward_function import FeatureBasedRewardFunction, State, BaseRewardFunction
from itertools import accumulate
from irl_benchmark.config import IRL_CONFIG_DOMAINS, IRL_ALG_REQUIREMENTS
from irl_benchmark.rl.algorithms.value_iteration import ValueIteration
from irl_benchmark.rl.algorithms.base_algorithm import BaseRLAlgorithm
from irl_benchmark.utils.wrapper import unwrap_env
from irl_benchmark.irl.reward.reward_wrapper import RewardWrapper

from cvxopt import solvers


class LinProgIRL(BaseIRLAlgorithm):
    "Linear Programming IRL (Ng & Russell)"

    def __init__(
        self,
        env: gym.Env,
        expert_trajs: List[Dict[str, list]],
        rl_alg_factory: Callable[[gym.Env], BaseRLAlgorithm],
        metrics: List[BaseMetric],
        config: dict,
    ):

        super(LinProgIRL, self).__init__(
            env, expert_trajs, rl_alg_factory, metrics, config
        )

    def train(
        self,
        no_irl_iterations: int,
        no_rl_episodes_per_irl_iteration: int,
        no_irl_episodes_per_irl_iteration: int,
    ) -> Tuple[BaseRewardFunction, BaseRLAlgorithm]:
        """Train the Linear Programming IRL algorithm. """

        # Step 0: Randomly initialize the reward function.

        reward_function = FeatureBasedRewardFunction(self.env, parameters="random")

        # Step 1: Get empirical total discounted return for the expert trajectories based on this reward function
        # empirical_total_discounted_reward = empirical_return(trajectories, reward_function, self.config['gamma'])

        # Step 2: Start with an abitrary policy

        policy = np.random.rand(16, 4)
        policy /= np.sum(policy, axis=1).reshape((-1, 1))

        # Step 3: Loop

        expert_state_only_trajs = [traj["states"] for traj in self.expert_trajs]
        expert_coeffs = trajs_to_value_coeffs(expert_state_only_trajs)
        policies = [policy]

        coeffs_diff_list = []

        agent = self.rl_alg_factory(self.env)
        reward_wrapper = unwrap_env(self.env, RewardWrapper)
        # params = reward_wrapper.reward_function.parameters

        for i in range(100):

            policy = policies[-1]

            trajs = generate_trajs_from_policy(self.env, policy)
            coeffs = trajs_to_value_coeffs(trajs)

            coeffs_diff_list.append(coeffs - expert_coeffs)

            # breakpoint()
            c = np.sum(coeffs_diff_list, axis=0)

            A = np.eye(16)
            b = np.ones(16)

            sol = solvers.lp(c, A, b)

            alphas = sol["x"]

            reward_wrapper.update_reward_parameters(alphas)
            agent.train(no_rl_episodes_per_irl_iteration=1000)
            next_policy = agent.policy_array()

            policies.append(policy)

        return alphas


def trajs_to_value_coeffs(trajs):

    # takes in trajs as list of states
    coeffs = np.zeros(16)
    # calculate value of trajs for each
    for traj in trajs:
        # traj is list of states visited
        for idx, state in enumerate(traj):
            coeffs[state] += .9** idx # TODO: remove hardcoded gamma value

    coeffs /= len(trajs)

    return coeffs


def generate_trajs_from_policy(env, policy, no_of_trajs=100):

    trajs = []
    for i in range(no_of_trajs):
        s0 = env.reset()
        states = [s0]
        done = False
        current_state = s0
        while not done:
            action = np.argmax(policy[current_state])
            current_state, _, done, _ = env.step(action)
            states.append(current_state)
        trajs.append(states)

    return trajs


def empirical_return(trajectories, reward_function, gamma):
    """Fast computation of empirical return averaged over all trajectories."""

    empirical_returns = []
    for traj in trajectories:
        states = State()
        rewards = reward_function(states)
        reversed_rewards = rewards[::-1]  # list reversal
        acc = list(accumulate(reversed_rewards, lambda x, y: x * gamma + y))
        total_discounted_reward = np.sum(acc)
        empirical_returns.append(total_discounted_reward)
    avg_emp_ret = np.mean(empirical_returns)

    return emp_ret


def p(x, _lambda=2):
    if x >= 0:
        return x
    else:
        return _lambda * x


IRL_CONFIG_DOMAINS[LinProgIRL] = {
    "gamma": {"type": float, "min": 0.0, "max": 1.0, "default": 0.9},
    "_lambda": {"type": float, "min": 0.0, "max": 10.0, "default": 2.0},
}

IRL_ALG_REQUIREMENTS[LinProgIRL] = {
    "requires_features": False,
    "requires_transitions": False,  # TODO: check again
}
