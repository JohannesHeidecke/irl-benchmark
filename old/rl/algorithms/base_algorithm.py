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

    def train(self, time_limit, metrics_listener=None):
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
        '''Return an optimal policy distribution.'''
        pass

    def joint_prob_of_actions(self, traj):
        '''Return joint probability of traj.

        This is the function U from Boularias et al. (2011, p. 185),
        which they describe as 'the joint probability of the actions
        conditioned on the states in [traj]'.
        '''
        return self.policy_prob_of_actions(traj)

    def policy_prob_of_actions(self, traj):
        '''Return product of the policy's action probabilities for traj.'''
        states = traj['states']
        acts = traj['actions']
        # probs = [self.policy(states[i])[acts[i]] for i in range(len(acts))]
        return 1  # np.prod(np.array(probs))


class SLMAlgorithm(RLAlgorithm):
    '''Parent class for agents from the SLM-Lab RL Library.

    We're using a custom fork of SLM-Lab:
    https://github.com/JohannesHeidecke/SLM-Lab

    We might eventually open a pull request to the original SLM-Lab:
    https://github.com/kengz/SLM-Lab
    '''

    def __init__(self, env, spec_file, spec_name):
        '''Set environment and get spec JSON file from spec_file.'''

        self.env = env
        self.spec = spec_util.get(spec_file=spec_file, spec_name=spec_name)
        # make sure the spec only contains one agent and one environment:
        assert len(self.spec['agent']) == 1
        assert len(self.spec['env']) == 1
        # make sure the specification is for the right environment id:
        self.spec['env'][0]['name'] = env.spec.id
        self.info_space = InfoSpace()
        # self.agent will only be filled after training.
        self.agent = None

        os.environ['PREPATH'] = slm_util.get_prepath(self.spec,
                                                     self.info_space)

    def train(self, time_limit):
        '''Create a SLM-lab session to train for time_limit seconds.'''
        os.environ['lab_mode'] = 'training'
        session = Session(self.spec, self.info_space)
        session.update_env(self.env)
        _, agent = session.run(time_limit)
        # persist trained agent:
        self.agent = agent


class RandomAgent(RLAlgorithm):
    '''Agent picking actions uniformly at random, for test purposes.

    Only works in environments with finite action space.
    Must be initialized with gym.Env environment.
    '''
    def pick_action(self, state):
        return self.env.action_space.sample()

    def policy(self, state):
        return np.ones(self.env.action_space.n) / self.env.action_space.n
