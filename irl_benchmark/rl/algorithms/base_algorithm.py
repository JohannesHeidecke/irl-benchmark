import gym
import numpy as np
import os

from slm_lab.experiment.monitor import InfoSpace
from slm_lab.spec import spec_util
from slm_lab.lib import util as slm_util

from irl_benchmark.rl.slm_session import Session


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

    def __init__(self, env, spec_file, spec_name):
        '''Set environment and get spec JSON file from spec_file.'''

        self.spec = spec_util.get(spec_file=spec_file, spec_name=spec_name)
        self.info_space = InfoSpace()
        self.agent = None
        
        os.environ['PREPATH'] = slm_util.get_prepath(self.spec, self.info_space)

    def train(self, time_limit):
        os.environ['lab_mode'] = 'training'
        # TODO: there should be a time limit here, not episode limit!
        self.spec['env'][0]['max_episode'] = 10

        session = Session(self.spec, self.info_space)
        _, agent = session.run()
        self.agent = agent



class RandomAgent(RLAlgorithm):
    '''Agent picking actions uniformly at random, for test purposes.

    Only works in environments with finite action space.
    '''

    def pick_action(self, state):
        assert isinstance(self.env.action_space, gym.spaces.Discrete)
        return np.random.choice(self.env.action_space.n)
