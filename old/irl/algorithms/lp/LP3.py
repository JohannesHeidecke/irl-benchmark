from irl_benchmark.irl.algorithms.base_algorithm import BaseIRLAlgorithm


class LP3(BaseIRLAlgorithm):

    def __init__(self, env, expert_trajs, rl_alg_factory, violated_constraint_weight=2):
        '''3rd linear programming (LP) alg from Ng & Russell (2000).

        Args:
          env: environment
          expert_trajs: expert trajectories
          violated_constraint_weight: controls penalty for violating the
            constraint that the expert perform better than the policies based
            on the IRL algorithm's intermediate reward estimates; Ng & Russell
            (p. 6) say the results aren't extremely sensitive to the value of
            this weight, and that they've heuristically chosen 2 as its value
        '''
        super(LP3, self).__init__(
            env=env,
            expert_trajs=expert_trajs,
            rl_alg_factory=rl_alg_factory,
        )
