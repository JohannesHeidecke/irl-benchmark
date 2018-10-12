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

        self.P = get_transition_matrix(env)
        self.rewards = None
        self.n_actions = env.action_space.n

    def mellowmax(self, q):
    '''
    This paper reports mellowmax has desirable properties that softmax doesn't.

    https://arxiv.org/pdf/1612.05628.pdf
    :param values: 1-D numpy array :TODO: check if we need for higher dimensional
    :param omega: the mellowmax parameter
    '''
    t = self.temperature
    return np.log(np.sum(np.exp(t * values))) - np.log(len(values)) / t

    def _boltzmann_vi(self, time_limit, metrics_listener=None):
        t0 = time()

        self.rewards = get_reward_matrix(self.env)

        n_states, n_actions, _ = np.shape(self.P)

        values = np.zeros([n_states])
        qs = np.zeros([n_states, n_actions])

        err = float('inf')

        # estimate values
        while err > self.error:
            values_old = values.copy()

            for s in range(n_states):
                # Get the q values for this state
                q = np.array([
                    sum([
                        self.P[s, a, s1] *
                        (self.rewards[s, a] + self.gamma * values[s1])
                        for s1 in range(n_states)
                    ]) for a in range(n_actions)
                ])
                qs[s] = q
                values[s] = self.mellowmax(q)

            err = np.max(np.abs(values - values_old))

            if time() > t0 + time_limit:
                print('Value iteration exceeded time limit with max err = {}'.
                      format(err))
                break

        # generate policy
        policy = np.zeros([n_states, n_actions])
        for s in range(n_states):
            # Get the q values for this state
            q = np.array([
                sum([
                    self.P[s, a, s1] *
                    (self.rewards[s, a] + self.gamma * values[s1])
                    for s1 in range(n_states)
                ]) for a in range(n_actions)
            ])
            best_actions = q == np.max(q)
            n_best = best_actions.sum()

            policy[s, best_actions] = 1.0 / n_best

        self.V = values
        self.pi = policy

    def train(self, time_limit, metrics_listener=None, reward_function=None):
        # TODO(ao) move this somewhere else
        if reward_function is not None:
            for state in range(self.rewards.shape[0]):
                for action in range(self.rewards.shape[1]):
                    self.rewards[state] = reward_function[state]

        if self.temperature is not None:
            return self._boltzmann_vi(time_limit, metrics_listener)
        t0 = time()

        n_states, n_actions, _ = np.shape(self.P)

        values = np.zeros([n_states])

        err = float('inf')

        # estimate values
        while err > self.error:
            values_old = values.copy()

            for s in range(n_states):
                values[s] = max([
                    sum([
                        self.P[s, a, s1] *
                        (self.rewards[s, a] + self.gamma * values_old[s1])
                        for s1 in range(n_states)
                    ]) for a in range(n_actions)
                ])

            err = np.max(np.abs(values - values_old))

            if time() > t0 + time_limit:
                print('Value iteration exceeded time limit with max err = {}'.
                      format(err))
                break

        # generate policy
        policy = np.zeros([n_states, n_actions])
        for s in range(n_states):
            # Get the q values for this state
            q = np.array([
                sum([
                    self.P[s, a, s1] *
                    (self.rewards[s, a] + self.gamma * values[s1])
                    for s1 in range(n_states)
                ]) for a in range(n_actions)
            ])
            best_actions = q == np.max(q)
            n_best = best_actions.sum()

            policy[s, best_actions] = 1.0 / n_best

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
