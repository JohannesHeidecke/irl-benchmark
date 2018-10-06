class BaseIRLAlgorithm(object):

    def __init__(self, env):
        self.env = env

    def train(self):
        raise NotImplementedError()

    def reward_function(self):
        ''' Returns the current reward function estimate '''
        raise NotImplementedError()