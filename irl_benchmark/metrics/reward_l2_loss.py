from irl_benchmark.irl.reward_function import RewardFunction


class RewardL2Loss(object):
    '''L2 distance between true reward and closest shaping of estimated reward.

    This is a metric not used in the literature so far.

    Many different reward functions lead to the same ordering of policies.
    It therefore doesn't make sense to look at the L2 distance of individual
    reward functions. Instead, we rely on Ng, Harada & Russell's (1999)
    characterization of the equivalence class of rewards inducing the same
    optimal policy in all MDPs. We search for the reward in the equivalence
    class of the IRL reward output that minimizes the L2 distance to the true
    reward and return that distance.
    '''
    def __init__(self, true_reward: RewardFunction):
        '''Pass true reward.'''
        self.true_reward = true_reward

    def evaluate(self, estim_reward: RewardFunction, use_sampling=False) -> floats:
        '''Return distance between estim_reward and true reward.'''
        # find best transformation from estim_reward to true_reward:
        # fitted_estim_reward = ...

        # return ||true_reward - fitted_estim_reward||_2
