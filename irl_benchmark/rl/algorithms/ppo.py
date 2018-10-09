from irl_benchmark.rl.algorithms import SLMAlgorithm

class PPO(SLMAlgorithm):

    def __init__(self, env, spec_file='ppo.json', spec_name='ppo_mlp_shared_pendulum'):
        super(PPO, self).__init__(
            env=env,
            spec_file=spec_file,
            spec_name=spec_name
        )

    def pick_action(self, state):
        action, _ = self.agent.algorithm.action_policy(state, self, body)
