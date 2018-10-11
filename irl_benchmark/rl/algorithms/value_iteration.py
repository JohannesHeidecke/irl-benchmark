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
            error=0.01,
    ):
        self.env = env
        self.gamma = gamma
        self.error = error

        self.P = get_transition_matrix(env)
        self.rewards = get_reward_matrix(env)
        self.n_actions = env.action_space.n

    def train(self, time_limit, metrics_listener=None):
        t0 = time()

        n_states, n_actions, _ = np.shape(self.P)

        values = np.zeros([n_states])

        # estimate values
        while True:
            values_tmp = values.copy()

            for s in range(n_states):
                values[s] = max([
                    sum([
                        self.P[s, a, s1] *
                        (self.rewards[s, a] + self.gamma * values_tmp[s1])
                        for s1 in range(n_states)
                    ]) for a in range(n_actions)
                ])

            err = np.max(np.abs(values - values_tmp))
            if err < self.error:
                break
            if time() > t0 + time_limit:
                print('Value iteration exceeded time limit with max err = {}'.
                      format(err))
                break

        # generate deterministic policy
        policy = np.zeros([n_states, n_actions])
        for s in range(n_states):
            a = np.argmax([
                sum([
                    self.P[s, a, s1] *
                    (self.rewards[s, a] + self.gamma * values[s1])
                    for s1 in range(n_states)
                ]) for a in range(n_actions)
            ])
            policy[s, a] = 1.0

        self.V = values
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
