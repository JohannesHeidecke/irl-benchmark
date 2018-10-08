class BaseIRLAlgorithm(object):

    def __init__(self, env, expert_trajs):
        self.env = env
        self.expert_trajs = expert_trajs

    def train(self, time_limit=300):
        raise NotImplementedError()

    def reward_function(self):
        ''' Returns the current reward function estimate '''
        raise NotImplementedError()