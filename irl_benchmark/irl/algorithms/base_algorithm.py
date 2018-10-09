class BaseIRLAlgorithm(object):

    def __init__(self, env, expert_trajs, rl_alg_factory):
        """The (abstract) base class for all IRL algorithms.

        Args:
          env : gym environment
          expert_trajs : list of expert trajectories (see irl/collect)
        """
        self.env = env
        self.expert_trajs = expert_trajs
        self.rl_alg_factory = rl_alg_factory

    def train(self, time_limit=300):
        """Train up to time_limit seconds."""
        raise NotImplementedError()

    def get_reward_function(self):
        """Return the current reward function estimate."""
        raise NotImplementedError()
