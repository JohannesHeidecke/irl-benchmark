import numpy as np

from irl_benchmark.irl.reward.reward_function import AbstractRewardFunction
from irl_benchmark.rl.algorithms.value_iteration import ValueIteration


class ILE():

    '''Follows formulation of Inverse Learning Error from Arora and Doshi (2018)'''

    def __init__(self, env, true_reward, gamma):
        self.env = env
        self.true_reward = true_reward
        self.gamma= gamma

    def evaluate(self, estim_rewards, ord=2):

        # value function for actual policy

        expert_agent = ValueIteration(env= self.env, gamma= self.gamma)
        expert_agent.train(time_limit=50)
        value_actual = expert_agent.V

        # value function for learned policy

        learned_agent = ValueIteration(env=self.env, gamma= self.gamma)
        learned_agent.train(time_limit=50, reward_function=estim_rewards)
        value_learned = learned_agent.V

        # Inverse Learning Error

        ile = np.linalg.norm((value_actual - value_learned), ord=ord)

        return ile

