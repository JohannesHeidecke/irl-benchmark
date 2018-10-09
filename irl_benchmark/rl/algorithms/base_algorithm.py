import gym
import numpy as np

from slm_lab.experiment.monitor import InfoSpace
from slm_lab.spec import spec_util


class RLAlgorithm(object):
    '''Base class for Reinforcement Learning agents.'''

    def __init__(self, env):
        '''Set environment.'''
        self.env = env

    def train(self, time_limit):
        '''Train for at most time_limit seconds.'''
        pass

    def save(self, path):
        '''Save agent parameters to path.'''
        pass

    def load(self, path):
        '''Load agent parameters from path.'''
        pass

    def pick_action(self, state, training=False):
        '''Return action to be taken in state.'''
        pass

    def policy(self, state):
        """ Return an optimal policy distribution """
        pass


class SLMAlgorithm(RLAlgorithm):
    '''Parent class for agents from the SLM-Lab RL Library.

    We're using a custom fork of SLM-Lab:
    https://github.com/JohannesHeidecke/SLM-Lab

    We might eventually open a pull request to the original SLM-Lab:
    https://github.com/kengz/SLM-Lab
    '''

    def __init(self, env, spec_file, spec_name):
        '''Set environment and get spec JSON file from spec_file.'''
        spec = spec_util.get(
            spec_file='frozen.json', spec_name='ddqn_epsilon_greedy_frozen')
        info_space = InfoSpace()
        # TODO: enable RewardWrapper envs in slm lab!!
        # finish this


class RandomAgent(RLAlgorithm):
    '''Agent picking actions uniformly at random, for test purposes.

    Only works in environments with finite action space.
    '''

    def pick_action(self, state):
        assert isinstance(self.env.action_space, gym.spaces.Discrete)
        return np.random.choice(self.env.action_space.n)
