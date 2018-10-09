from irl_benchmark.irl.algorithms.base_algorithm import BaseIRLAlgorithm
from irl_benchmark.irl.collect import collect_trajs
from irl_benchmark.irl.reward.reward_function import FeatureBasedRewardFunction
from irl_benchmark.rl.algorithms import RandomAgent, TabularQ


class MaxEnt(BaseIRLAlgorithm):

    """Maximum Entropy IRL (Ziebart et al., 2008).

    Not to be confused with Maximum Entropy Deep IRL (Wulfmeier et al., 2016)
    or Maximum Causal Entropy IRL (Ziebart et al., 2010).
    """

    def __init__(self, env, expert_trajs, gamma = .99):
        super(MaxEnt, self).__init__(env, expert_trajs)
        self.gamma = gamma



    def expected_svf(self, ):