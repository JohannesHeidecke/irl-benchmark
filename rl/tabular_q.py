import numpy as np
from collections import defaultdict

EPS = 1e-100

class TabularQ:

    def __init__(self, env, alpha=0.8, gamma=0.95, start_eps=0.9, end_eps=0.05, eps_decay=1e-5):
        self.alpha = alpha
        self.gamma = gamma

        self.start_eps = start_eps
        self.end_eps = end_eps
        self.eps_decay = eps_decay

        self.n_actions = env.action_space.n

        self.Q = defaultdict(lambda: np.zeros(self.n_actions))
        self.n_updates = 0

    def get_eps(self):
        """ Returns the annealed value of exploration amount. """
        return max(self.end_eps, self.start_eps - self.n_updates*self.eps_decay)

    def act(self, s):
        if np.random.rand() < self.get_eps():
            return np.random.randint(0, self.n_actions)
        return np.argmax(self.Q[s])

    def update(self, s, a, r, sp):
        Q = self.Q[s][a]
        Qp = np.max(self.Q[sp])
        self.Q[s][a] += self.alpha*(r + self.gamma*Qp - Q)
        self.n_updates += 1
