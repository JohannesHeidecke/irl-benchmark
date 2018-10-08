from irl_benchmark.irl.algorithms.base_algorithm import BaseIRLAlgorithm


class LP3(BaseIRLAlgorithm):

    def __init__(self, env, expert_trajs, violated_constraint_weight=2):
        super(LP3, self).__init__(env)
