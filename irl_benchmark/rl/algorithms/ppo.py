'''Proximal Policy Optimization'''
from gym.spaces import Box
import numpy as np

from irl_benchmark.rl.algorithms import SLMAlgorithm


class PPO(SLMAlgorithm):

    def __init__(self, env, spec_file='ppo.json',
                 spec_name='ppo_mlp_shared_pendulum'):
        super(PPO, self).__init__(
            env=env,
            spec_file=spec_file,
            spec_name=spec_name
        )

    def pick_action(self, state):
        action = self.agent.algorithm.act(state)
        if isinstance(self.env.action_space, Box) and np.isscalar(action):
            # Box action spaces require even single actions to be a list:
            action = [action]
        return action
