'''Proximal Policy Optimization'''
from gym.spaces import Box
from gym.spaces import Discrete as DiscreteSpace
import numpy as np

from irl_benchmark.rl.algorithms import SLMAlgorithm
from irl_benchmark.utils.utils import to_one_hot


class PPO(SLMAlgorithm):

    def __init__(self, env, spec_file='ppo.json',
                 spec_name='ppo_mlp_shared_pendulum'):
        super(PPO, self).__init__(
            env=env,
            spec_file=spec_file,
            spec_name=spec_name
        )

    def pick_action(self, state):
        if isinstance(self.env.observation_space, DiscreteSpace):
            # state needs to be one-hot encoded to be suitable neural network input
            state = to_one_hot(state, self.env.observation_space.n)
        action = self.agent.algorithm.act(state)
        if isinstance(self.env.action_space, Box) and np.isscalar(action):
            # Box action spaces require even single actions to be a list:
            action = [action]
        return action
