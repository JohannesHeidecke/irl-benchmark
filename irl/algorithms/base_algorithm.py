class BaseIRLAlgorithm(object):
    """ The (abstract) base class for all IRL algorithms.
    """

    def __init__(self, env, expert_trajs):
        """ 
        Parameters
        ----------
        env : gym environment
        expert_trajs : list of expert trajectories (see irl/collect)
        """
        self.env = env
        self.expert_trajs = expert_trajs

    def train(self, time_limit=300):
        """ Train the algorithm up to time_limit seconds. """
        raise NotImplementedError()

    def get_reward_function(self):
        """ Returns the current reward function estimate """
        raise NotImplementedError()