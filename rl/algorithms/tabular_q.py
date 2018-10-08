import numpy as np
from collections import defaultdict
from time import time
import pickle
from rl.algorithm import RLAlgorithm


class TabularQ(RLAlgorithm):
    """ 
    Tabular Q algorithm.
    """

    def __init__(
        self,
        env,
        gamma=0.95,
        alpha_start=0.8,
        alpha_end=0.1,
        alpha_decay=1e-4,
        eps_start=0.9,
        eps_end=0.01,
        eps_decay=1e-4,
    ):
        self.env = env
        self.gamma = gamma

        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.alpha_decay = alpha_decay

        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay

        self.n_actions = env.action_space.n

        self.Q = defaultdict(self.init_q)
        self.n_updates = 0

    def init_q(self):
        return np.zeros(self.n_actions)

    def get_alpha(self):
        """ Returns the annealed value of learning rate. """
        return max(self.alpha_end, self.alpha_start - self.n_updates*self.alpha_decay)

    def get_eps(self):
        """ Returns the annealed value of exploration amount. """
        return max(self.eps_end, self.eps_start - self.n_updates*self.eps_decay)

    def pick_action(self, s):
        if np.random.rand() < self.get_eps():
            return np.random.randint(0, self.n_actions)
        return np.argmax(self.Q[s])

    def update(self, s, a, r, sp):
        Q = self.Q[s][a]
        Qp = np.max(self.Q[sp])
        self.Q[s][a] += self.get_alpha()*(r + self.gamma*Qp - Q)
        self.n_updates += 1

    def train(self, time_limit):
        t0 = time()

        sum_rewards = []
        while time() < t0 + time_limit:
            rs = []
            s = self.env.reset()
            done = False

            while not done:
                a = self.pick_action(s)
                sp, r, done, info = self.env.step(a)
                self.update(s, a, r, sp)
                s = sp
                rs.append(r)

            sum_reward = np.sum(rs)
            sum_rewards.append(sum_reward)

        return sum_rewards

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def load(self, path):
        """" Loads in instance of self and copies attributes. """
        with open(path, 'rb') as f:
            parsed = pickle.load(f)
        for k, v in parsed.__dict__.items():
            setattr(self, k, v)

