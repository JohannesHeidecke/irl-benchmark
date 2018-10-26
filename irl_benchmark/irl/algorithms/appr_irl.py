"""Module for apprenticeship learning IRL."""
from typing import Callable, Dict, List, Tuple

import cvxpy as cvx
import gym
import numpy as np

from irl_benchmark.config import IRL_CONFIG_DOMAINS
from irl_benchmark.irl.algorithms.base_algorithm import BaseIRLAlgorithm
from irl_benchmark.irl.collect import collect_trajs
from irl_benchmark.irl.reward.reward_function import BaseRewardFunction, FeatureBasedRewardFunction
from irl_benchmark.rl.algorithms.base_algorithm import BaseRLAlgorithm
from irl_benchmark.rl.algorithms.random_agent import RandomAgent
from irl_benchmark.utils.rl import true_reward_per_traj


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
            The fields are:
            * 'gamma': discount factor between 0. and 1.
            * 'epsilon': small positive value, stopping criterion.
            * 'mode': which variant of the algorithm to use, either 'svm' or 'projection'.
        """
        super(ApprIRL, self).__init__(env, expert_trajs, rl_alg_factory,
                                      config)

        # calculate the feature counts of expert trajectories:
        self.expert_feature_count = self.feature_count(
            self.expert_trajs, self.config['gamma'])

        # create list of feature counts:
        self.feature_counts = [self.expert_feature_count]
        # for SVM mode: create list of labels:
        self.labels = [1.]

        self.distances = []

    def train(self, no_irl_iterations: int,
              no_rl_episodes_per_irl_iteration: int,
              no_irl_episodes_per_irl_iteration: int
              ) -> Tuple[BaseRewardFunction, BaseRLAlgorithm]:

        # TODO: replace all prints with adequate logger output
        # TODO: add docstring and comments

        agent = RandomAgent(self.env)

        irl_iteration_counter = 0
        while irl_iteration_counter < no_irl_iterations:
            irl_iteration_counter += 1

            if self.config['verbose']:
                print('IRL ITERATION ' + str(irl_iteration_counter))

            trajs = collect_trajs(
                self.env, agent, no_trajectories=no_irl_episodes_per_irl_iteration)

            if self.config['verbose']:
                print('Average true reward per episode: ' +
                      str(true_reward_per_traj(trajs)))

            current_feature_count = self.feature_count(trajs, gamma=self.config['gamma'])
            self.feature_counts.append(current_feature_count)
            self.labels.append(-1.)

            feature_counts = np.array(self.feature_counts)
            labels = np.array(self.labels)

            if self.config['mode'] == 'projection':
                if irl_iteration_counter == 1:
                    feature_count_bar = feature_counts[1]
                else:
                    line = feature_counts[-1] - feature_count_bar
                    feature_count_bar += np.dot(
                        line, feature_counts[0] - feature_count_bar) / np.dot(
                            line, line) * line
                reward_coefficients = feature_counts[0] - feature_count_bar
                distance = np.linalg.norm(reward_coefficients)

            elif self.config['mode'] == 'svm':
                w = cvx.Variable(feature_counts.shape[1])
                b = cvx.Variable()

                objective = cvx.Minimize(cvx.norm(w, 2))
                constraints = [
                    cvx.multiply(labels, (feature_counts * w + b)) >= 1
                ]

                problem = cvx.Problem(objective, constraints)
                problem.solve()

                if w.value is None:
                    # TODO: replace by logger warning
                    print('NO MORE SVM SOLUTION!!')
                    return

                yResult = feature_counts.dot(w.value) + b.value
                supportVectorRows = np.where(np.isclose(np.abs(yResult), 1))[0]

                if self.config['verbose']:
                    print('The support vectors are from iterations number ' +
                          str(supportVectorRows))

                reward_coefficients = w.value
                distance = 2 / problem.value

            else:
                raise NotImplementedError()

            if self.config['verbose']:
                print('Reward coefficients: ' + str(reward_coefficients))
                print('Distance: ' + str(distance))

            self.distances.append(distance)

            reward_function = FeatureBasedRewardFunction(
                self.env, reward_coefficients)
            self.env.update_reward_function(reward_function)

            if distance <= self.config['epsilon']:
                if self.config['verbose']:
                    print("Feature counts matched within " +
                          str(self.config['epsilon']) + ".")
                break

            agent = self.rl_alg_factory(self.env)
            agent.train(no_rl_episodes_per_irl_iteration)

        return reward_function, agent


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
    'verbose': {
        'type': bool,
        'default': True
    }
}
