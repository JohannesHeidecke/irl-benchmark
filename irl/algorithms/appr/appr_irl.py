import cvxpy as cvx
import numpy as np
import time

from irl.algorithms.base_algorithm import BaseIRLAlgorithm
from irl.collect import collect_trajs
from irl.reward.reward_function import FeatureBasedRewardFunction
from rl.algorithms import RandomAgent, TabularQ

class ApprIRL(BaseIRLAlgorithm):

    def __init__(self, env, expert_trajs, gamma=0.99, proj=False):
        super(ApprIRL, self).__init__(env, expert_trajs)
        self.gamma = gamma
        self.proj = proj  # Projection alg if true, else max-margin.

        self.expert_feature_count = self.feature_count(expert_trajs)
        self.feature_counts = [self.expert_feature_count]
        self.labels = [1.0]

        self.distances = []

    def train(self, time_limit=300, rl_time_per_iteration=30, eps=0, verbose=False):
        '''Accumulate feature counts and estimate reward function.

        Args:
          time_limit: total training time in seconds
          rl_time_per_iteration: RL training time per step

        Returns nothing.
        '''
        t0 = time.time()

        if verbose:
            alg_mode = 'projection' if self.proj else 'SVM'
            print('Running Apprenticeship IRL in mode: ' + alg_mode)
            print(time_limit)
            print(rl_time_per_iteration)

        # start with random agent:
        agent = RandomAgent(self.env)

        iteration_counter = 0
        while time.time() < t0 + time_limit:
            iteration_counter += 1
            if verbose:
                print('ITERATION ' + str(iteration_counter))
            trajs = collect_trajs(self.env, agent, no_episodes=100,
                                  max_steps_per_episode=100)

            current_feature_count = self.feature_count(trajs)
            self.feature_counts.append(current_feature_count)
            self.labels.append(-1.0)

            feature_counts = np.array(self.feature_counts)
            labels = np.array(self.labels)

            if self.proj:
                # using projection version of the algorithm
                if iteration_counter == 1:
                    feature_count_bar = feature_counts[1]
                else:
                    line = feature_counts[-1] - feature_count_bar
                    feature_count_bar += np.dot(line, feature_counts[0] - feature_count_bar) / np.dot(line, line) * line
                reward_coefficients = feature_counts[0] - feature_count_bar
                distance = np.linalg.norm(reward_coefficients)
                            
            else:
                # using SVM version of the algorithm
                w = cvx.Variable(feature_counts.shape[1])
                b = cvx.Variable()

                objective = cvx.Minimize(cvx.norm(w, 2))
                constraints = [cvx.multiply(labels, (feature_counts * w + b)) >= 1]

                problem = cvx.Problem(objective, constraints)
                problem.solve()

                yResult = feature_counts.dot(w.value) + b.value
                supportVectorRows = np.where(np.isclose(np.abs(yResult), 1))[0]

                reward_coefficients = w.value
                distance = 2/problem.value

                if verbose:
                    print('The support vectors are from iterations number ' +
                          str(supportVectorRows))
            if verbose:
                print('Reward coefficients: ' + str(reward_coefficients))
                print('Distance: ' +str(distance))

            self.distances.append(distance)

            self.reward_function = FeatureBasedRewardFunction(self.env, reward_coefficients)
            self.env.update_reward_function(self.reward_function)

            if distance <= eps:
                if verbose:
                    print("Feature counts matched within " + str(eps) + ".")
                break

            if time.time() + rl_time_per_iteration >= t0 + time_limit:
                break

            agent = TabularQ(self.env)
            agent.train(rl_time_per_iteration)

    def get_reward_function(self):
        return self.reward_function

    def feature_count(self, trajs):
        '''Calculate empirical feature counts of input trajectories.'''
        feature_sum = np.zeros(self.env.env.feature_shape())
        for traj in trajs:
            gammas = self.gamma ** np.arange(len(traj['features']))
            feature_sum += np.sum(
                gammas.reshape(-1, 1) * np.array(traj['features']), axis=0
            )
        feature_count = feature_sum / len(trajs)
        return feature_count
