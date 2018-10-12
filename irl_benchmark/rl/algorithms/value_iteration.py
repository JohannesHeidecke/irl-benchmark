import numpy as np
import pickle
from time import time

from irl_benchmark.rl.algorithms import RLAlgorithm
from irl_benchmark.utils.utils import get_transition_matrix, get_reward_matrix


class ValueIteration(RLAlgorithm):
    '''
    Solves an MDP optimally with value iteration.
    '''

    def __init__(
            self,
            env,
            gamma=0.8,
            error=1e-6,
            temperature=None,
    ):
        self.env = env
        self.gamma = gamma
        self.error = error
        self.temperature = temperature

        self.n_actions = env.action_space.n
        self.P = get_transition_matrix(env)
        self.rewards = None

    def softmax(self, q):
        '''∀ s: V_s = temperature * log(\sum_a exp(Q_sa/temperature))'''

        # We can rewrite as:
        # t*log(sum_a exp((q_a - qmax)/t)*exp(qmax/t))
        # subtracting q_max for robustness
        # qmax goes straight through below softmax function: t*log(exp(qmax/t)) = qmax

        q_max = q.max(axis=1, keepdims=True)
        scaled = (q - q_max) / self.temperature
        exp_sum = np.exp(scaled).sum(axis=1)

        values = self.temperature * np.log(exp_sum) + q_max.reshape([-1])
        return values

    def mellowmax(self, q):
        '''
        The below paper reports mellowmax has desirable properties that
        softmax doesn't.

        https://arxiv.org/pdf/1612.05628.pdf
        :param values: 2-D numpy array
        :param temperature: the softmax temperature
        '''

        softmax = self.softmax(q)
        return softmax - np.log(q.shape[1]) * self.temperature

    def _boltzmann_vi(self, time_limit, metrics_listener=None):
        t0 = time()

        n_states, n_actions, _ = np.shape(self.P)

        values = np.zeros([n_states])
        q = np.zeros([n_states, n_actions])

        err = float('inf')

        # estimate values
        while err > self.error:
            values_old = values.copy()

            q = self.gamma * self.P.dot(values) + self.rewards
            values = self.mellowmax(q)

            err = np.max(np.abs(values - values_old))

            if time() > t0 + time_limit:
                print('Value iteration exceeded time limit with max err = {}'.
                      format(err))
                break

        # Compute stochastic policy:

        # Broadcast values to shape of q
        self.pi = np.exp((q - values.reshape([-1, 1])) / self.temperature)

        # Mellowmax correction:
        self.pi = self.pi / q.shape[1]

        # Normalize:
        self.pi = self.pi / self.pi.sum(axis=1, keepdims=True)

        self.V = values
        self.Q = q

    def train(self, time_limit, metrics_listener=None, reward_function=None):
        self.rewards = get_reward_matrix(self.env)

        if self.temperature is not None:
            return self._boltzmann_vi(time_limit, metrics_listener)
        t0 = time()

        n_states, n_actions, _ = np.shape(self.P)

        values = np.zeros([n_states])
        q = np.zeros([n_states, n_actions])

        err = float('inf')

        # estimate values
        while err > self.error:
            values_old = values.copy()

            q = self.gamma * self.P.dot(values) + self.rewards
            values = q.max(axis=1)

            err = np.max(np.abs(values - values_old))

            if time() > t0 + time_limit:
                print('Value iteration exceeded time limit with max err = {}'.
                      format(err))
                break

        # generate policy
        policy = np.zeros([n_states, n_actions])

        for s in range(n_states):
            best_actions = q[s] == np.max(q[s])
            n_best = best_actions.sum()

            policy[s, best_actions] = 1.0 / n_best

        self.V = values
        self.Q = q
        self.pi = policy

    def policy(self, s):
        return self.pi[s]

    def pick_action(self, s):
        '''Sample an action from policy.'''
        return np.random.choice(range(self.n_actions), p=self.policy(s))

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
