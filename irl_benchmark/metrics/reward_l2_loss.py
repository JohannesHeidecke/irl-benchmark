from irl_benchmark.irl.reward_function import RewardFunction


class RewardL2Loss(object):

    def __init__(self, true_reward: RewardFunction):
        self.true_reward = true_reward

    def evaluate(self, estim_reward: RewardFunction, use_sampling=False) -> floats:

        # find best transformation from estim_reward to true_reward:
        # fitted_estim_reward = ...

        # return ||true_reward - fitted_estim_reward||_2
