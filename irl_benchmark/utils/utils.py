from collections import defaultdict
import json

import numpy as np
import pandas as pd

from gym.envs.toy_text.discrete import DiscreteEnv


def to_one_hot(hot_vals, max_val):
    '''Convert an int list of data into one-hot vectors.'''
    return np.eye(max_val)[np.array(hot_vals)]


def unwrap_env(env, until_class=None):
    '''
    Unwraps wrapped env until we get an instance that is a until_class.

    If cls is None we will unwrap all the way.
    '''
    if until_class is None:
        while hasattr(env, 'env'):
            env = env.env
        return env

    while hasattr(env, 'env') and not isinstance(env, until_class):
        env = env.env

    if not isinstance(env, until_class):
        raise ValueError(
            "Unwrapping env did not yield an instance of class {}".format(
                until_class))
    return env

def is_unwrappable_to(env, to_class):
    '''Check if env can be unwrapped to to_class.'''
    if isinstance(env, to_class):
        return True
    while hasattr(env, 'env'):
        env = env.env
        if isinstance(env, to_class):
            return True
    return False

def get_transition_matrix(env):
    '''Gets transition matrix from discrete environment.'''
    env = unwrap_env(env, DiscreteEnv)

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    table = np.zeros([n_states, n_actions, n_states])

    # Iterate over "from" states:
    for s, P_given_state in env.P.items():

        # Iterate over actions:
        for a, transitions in P_given_state.items():

            # Iterate over "to" states:
            for proba, sp, r, done in transitions:
                table[s, a, sp] += proba

    return table


def get_reward_matrix(env):
    '''Gets reward array from discrete environment.'''
    env = unwrap_env(env, DiscreteEnv)

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    rewards = np.zeros([n_states, n_actions])

    # Iterate over "from" states:
    for s, P_given_state in env.P.items():

        # Iterate over actions:
        for a, transitions in P_given_state.items():

            # Iterate over "to" states:
            for proba, sp, r, done in transitions:
                rewards[s, a] += r * proba

    return rewards


class MetricsLogger():
    '''
    Listens for metrics to be stored as json.
    Metrics can be stored once per run or once per training step.

    The simplest usage is to load the jsons of relevant runs,
    select metrics and convert to a pandas DataFrames.
    '''

    def __init__(self):
        self.metrics = defaultdict(lambda: [])

    def log_metric(self, name, value):
        self.metrics[name].append(value)

    def save(self, path):
        with open(path, 'wt') as f:
            json.dump(self.metrics, f)
