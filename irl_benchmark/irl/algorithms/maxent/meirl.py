from irl_benchmark.irl.algorithms.base_algorithm import BaseIRLAlgorithm


class MaxEnt(BaseIRLAlgorithm):
    '''Maximum Entropy IRL (Ziebart et al., 2008).

    Not to be confused with Maximum Entropy Deep IRL (Wulfmeier et al., 2016)
    or Maximum Causal Entropy IRL (Ziebart et al., 2010).
    '''
    def __init__(self, env, expert_trajs, rl_alg_factory):
        super(MaxEnt, self).__init__(
            env=env,
            expert_trajs=expert_trajs,
            rl_alg_factory=rl_alg_factory,
        )
