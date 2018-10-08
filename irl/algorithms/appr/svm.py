import cvxpy as cvx
import numpy as np
import time

from irl.algorithms.base_algorithm import BaseIRLAlgorithm
from irl.collect.collect_data import collect_trajs
from irl.reward.reward_function import FeatureBasedRewardFunction
from rl.algorithm import RandomAgent
from rl.tabular_q import TabularQ


class SVMIRL(BaseIRLAlgorithm):
    
    def __init__(self, env, expert_trajs, gamma=0.99):
        super(SVMIRL, self).__init__(env, expert_trajs)
        self.gamma = gamma

        self.expert_mu = self.mu(expert_trajs)
        self.mus = [self.expert_mu]
        self.labels = [1.0]


    def train(self, time_limit=300, rl_time_per_iteration=30):
       
        t0 = time.time()

        # start with random agent:
        agent = RandomAgent(self.env)

        iteration_counter = 0
        while time.time() < t0 + time_limit:
            iteration_counter += 1
            print('ITERATION ' + str(iteration_counter))
            trajs = collect_trajs(self.env, agent, no_episodes=100, max_steps_per_episode=100)

            current_mu = self.mu(trajs)
            self.mus.append(current_mu)
            self.labels.append(-1.0)

            mus = np.array(self.mus)
            labels = np.array(self.labels)

            w = cvx.Variable(mus.shape[1])
            b = cvx.Variable()

            objective   = cvx.Minimize(cvx.norm(w, 2))
            constraints = [cvx.multiply(labels, (mus * w + b)) >= 1]

            problem = cvx.Problem(objective, constraints)
            problem.solve()

            yResult =  mus.dot(w.value) + b.value
            supportVectorRows = np.where(np.isclose(np.abs(yResult), 1))[0]

            print('Problem status: ' + str(problem.status))
            print('Reward coefficients: ' + str(w.value))
            print('Reward bias: ' + str(b.value))
            print('Distance: ' + str(2/problem.value))
            print('The support vectors are mus number ' + str(supportVectorRows))
            print('(index 0 is expert demonstration, 1 random demonstration, higher: intermediate mus)')

            reward_function = FeatureBasedRewardFunction(self.env, w.value)
            self.env.update_reward_function(reward_function)

            agent = TabularQ(self.env)
            agent.train(rl_time_per_iteration)



    def mu(self, trajs):
        feature_sum = np.zeros(self.env.env.feature_shape())  
        for traj in trajs:
            gammas = self.gamma ** np.arange(len(traj['features']))
            feature_sum += np.sum(gammas.reshape(-1, 1) * np.array(traj['features']), axis=0)
        mu = feature_sum / len(trajs)
        return mu




    