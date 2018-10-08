import gym
import numpy as np

class RLAlgorithm(object):

    def __init__(self, env):
        self.env = env

    def train(self, time_limit):
        pass

    def save(self, path):
        pass

    def load(self, path):
        pass

    def pick_action(self, state):
        pass

    def policy(self):
        pass


class SLMAlgorithm(RLAlgorithm):

    def __init(self, env, spec_file, spec_name):
        spec = spec_util.get(spec_file='frozen.json', spec_name='ddqn_epsilon_greedy_frozen')
        info_space = InfoSpace()
        # TODO: enable RewardWrapper envs in slm lab!!
        # finish this 



class TabularQLearning(RLAlgorithm):
    pass

class RandomAgent(RLAlgorithm):

    def pick_action(self, state):
        assert isinstance(self.env.action_space, gym.spaces.Discrete)
        return np.random.choice(self.env.action_space.n)

