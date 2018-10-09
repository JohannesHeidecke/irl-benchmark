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

    def train(self, time_limit=300, rl_time_per_iteration=30):
        """Train up to time_limit seconds.

        Args:
          time_limit: total training time in seconds
          rl_time_per_iteration: RL training time per step in seconds.

        Returns nothing.
        """
        raise NotImplementedError()

    def get_reward_function(self):
        """Return the current reward function estimate."""
        raise NotImplementedError()
