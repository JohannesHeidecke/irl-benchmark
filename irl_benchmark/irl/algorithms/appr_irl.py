"""Module for apprenticeship learning IRL."""
from typing import Callable, Dict, List, Tuple

import cvxpy as cvx
import gym
import numpy as np

from irl_benchmark.config import IRL_CONFIG_DOMAINS
from irl_benchmark.irl.algorithms.base_algorithm import BaseIRLAlgorithm
from irl_benchmark.irl.collect import collect_trajs
from irl_benchmark.irl.reward.reward_function import BaseRewardFunction, FeatureBasedRewardFunction
from irl_benchmark.irl.reward.reward_wrapper import RewardWrapper
from irl_benchmark.metrics.base_metric import BaseMetric
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
                 metrics: List[BaseMetric], config: dict):
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
                                      metrics, config)

        # calculate the feature counts of expert trajectories:
        self.expert_feature_count = self.feature_count(self.expert_trajs,
                                                       self.config['gamma'])

        # create list of feature counts:
        self.feature_counts = [self.expert_feature_count]
        # for SVM mode: create list of labels:
        self.labels = [1.]

        self.distances = []

    def train(self, no_irl_iterations: int,
              no_rl_episodes_per_irl_iteration: int,
              no_irl_episodes_per_irl_iteration: int
              ) -> Tuple[BaseRewardFunction, BaseRLAlgorithm]:
        """Train the apprenticeship learning IRL algorithm.

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

        # Initialize training with a random agent.
        agent = RandomAgent(self.env)

        irl_iteration_counter = 0
        while irl_iteration_counter < no_irl_iterations:
            irl_iteration_counter += 1

            if self.config['verbose']:
                print('IRL ITERATION ' + str(irl_iteration_counter))

            # Estimate feature count of current agent.
            trajs = collect_trajs(
                self.env,
                agent,
                no_trajectories=no_irl_episodes_per_irl_iteration)
            current_feature_count = self.feature_count(
                trajs, gamma=self.config['gamma'])

            # add new feature count to list of feature counts
            self.feature_counts.append(current_feature_count)
            # for SVM mode:
            self.labels.append(-1.)

            # convert to numpy array:
            feature_counts = np.array(self.feature_counts)
            labels = np.array(self.labels)

            # update reward coefficients based on mode specified in config:
            if self.config['mode'] == 'projection':
                # projection mode:
                if irl_iteration_counter == 1:
                    # initialize feature_count_bar in first iteration
                    # set to first non-expert feature count:
                    feature_count_bar = feature_counts[1]
                else:
                    # not first iteration.
                    # calculate line through last feature_count_bar and
                    # last non-expert feature count:
                    line = feature_counts[-1] - feature_count_bar
                    # new feature_count_bar is orthogonal projection of
                    # expert's feature count onto the line:
                    feature_count_bar += np.dot(
                        line, feature_counts[0] - feature_count_bar) / np.dot(
                            line, line) * line
                reward_coefficients = feature_counts[0] - feature_count_bar
                # compute distance as L2 norm of reward coefficients (t^(i) in paper):
                distance = np.linalg.norm(reward_coefficients, ord=2)

            elif self.config['mode'] == 'svm':
                # svm mode:
                # create quadratic programming problem definition:
                weights = cvx.Variable(feature_counts.shape[1])
                bias = cvx.Variable()
                objective = cvx.Minimize(cvx.norm(weights, 2))
                constraints = [
                    cvx.multiply(labels,
                                 (feature_counts * weights + bias)) >= 1
                ]
                problem = cvx.Problem(objective, constraints)
                # solve quadratic program:
                problem.solve()

                if weights.value is None:
                    # TODO: we need to handle empty solution better.
                    raise RuntimeError(
                        'Empty solution set for linearly separable SVM.')

                if self.config['verbose']:
                    # print support vectors
                    # (which last iterations where relevant for current result?)
                    svm_classifications = feature_counts.dot(
                        weights.value) + bias.value
                    support_vectors = np.where(
                        np.isclose(np.abs(svm_classifications), 1))[0]
                    print('The support vectors are from iterations number ' +
                          str(support_vectors))

                reward_coefficients = weights.value
                distance = 2 / problem.value

            else:
                raise NotImplementedError()

            if self.config['verbose']:
                print('Distance: ' + str(distance))

            self.distances.append(distance)

            # create new reward function with current coefficient estimate
            reward_function = FeatureBasedRewardFunction(
                self.env, reward_coefficients)
            # update reward function
            assert isinstance(self.env, RewardWrapper)
            self.env.update_reward_function(reward_function)

            # TODO: see messages with max about order of training & deducing
            # check stopping criterion:
            if distance <= self.config['epsilon']:
                if self.config['verbose']:
                    print("Feature counts matched within " +
                          str(self.config['epsilon']) + ".")
                break

            # create new RL-agent
            agent = self.rl_alg_factory(self.env)
            # train agent (with new reward function)
            agent.train(no_rl_episodes_per_irl_iteration)

            evaluation_input = {
                'irl_agent': agent,
                'irl_reward': reward_function
            }
            self.evaluate_metrics(evaluation_input)

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
