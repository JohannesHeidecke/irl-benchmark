import numpy as np
from collections import defaultdict
from time import time
import pickle

from irl_benchmark.rl.algorithms import RLAlgorithm


class TabularQ(RLAlgorithm):
    '''Tabular Q-learning.

    Only works for discrete observation/action spaces.
    '''
    def __init__(
            self,
            env,
            gamma=0.8,
            alpha_start=0.8,
            alpha_end=0.1,
            alpha_decay=1e-4,
            eps_start=0.9,
            eps_end=0.01,
            eps_decay=1e-4,
    ):
        '''Set environment, discount rate, learning rate, and exploration rate.

        Learning and exploration rate decay linearly.

        Args:
          env: environment
          gamma: `float`, discount rate
          alpha_start: `float`, initial learning rate
          alpha_end: `float`, asymptotic learning rate
          alpha_decay: `float`, per-step decrease of learning rate
          eps_start: `float`, initial exploration rate
          eps_end: `float`, asymptotic exploration rate
          eps_decay: `float`, per-step decrease of exploration rate
        '''
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
        self.n_updates = 0  # No of decreases of learning and exploration rate.

    def init_q(self):
        '''Initialize Q-values for one state to zero.'''
        return np.zeros(self.n_actions)

    def get_alpha(self):
        '''Return the annealed value of learning rate.'''
        return max(self.alpha_end,
                   self.alpha_start - self.n_updates * self.alpha_decay)

    def get_eps(self):
        '''Return the annealed value of exploration amount.'''
        return max(self.eps_end,
                   self.eps_start - self.n_updates * self.eps_decay)

    def pick_action(self, s, training=False):
        '''If training return random action w/ prob eps, else act greedily.'''
        if training and np.random.rand() < self.get_eps():
            return np.random.randint(0, self.n_actions)
        return np.argmax(self.Q[s])

    def policy(self, s):
        a = self.pick_action(s)
        p = np.zeros(self.n_actions)
        p[a] = 1.0
        return p

    def update(self, s, a, r, sp):
        '''Update Q-values based on one transition.

        Args:
          s: `int`, previous state
          a: `int`, action taken in previous state
          r: `float`, reward received for transition
          sp: `int`, next state

        Return nothing.
        '''
        Q = self.Q[s][a]
        Qp = np.max(self.Q[sp])
        self.Q[s][a] += self.get_alpha() * (r + self.gamma * Qp - Q)
        self.n_updates += 1

    def train(self, time_limit, metric_listener=None):
        '''Train agent for at most time_limit seconds.

        Return undiscounted sum of rewards.'''
        t0 = time()

        sum_rewards = []
        while time() < t0 + time_limit:
            rs = []
            s = self.env.reset()
            done = False

            while not done:
                a = self.pick_action(s, training=True)
                sp, r, done, info = self.env.step(a)
                self.update(s, a, r, sp)
                s = sp
                rs.append(r)

            sum_reward = np.sum(rs)
            sum_rewards.append(sum_reward)

            # Push each episode to listener
            if metric_listener is not None:
                metric_listener(sum_reward)

        return sum_rewards

    def save(self, path):
        '''Save agent parameters to path.'''
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def load(self, path):
        '''Load in instance of self and copy attributes.'''
        with open(path, 'rb') as f:
            parsed = pickle.load(f)
        for k, v in parsed.__dict__.items():
            setattr(self, k, v)
