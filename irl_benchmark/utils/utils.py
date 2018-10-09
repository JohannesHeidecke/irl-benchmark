import json
from collections import defaultdict

import numpy as np
from gym.envs.toy_text.discrete import DiscreteEnv

import irl_benchmark.irl.reward.reward_wrapper as rew_wrapper
import irl_benchmark.irl.feature.feature_wrapper as feat_wrapper


def to_one_hot(hot_vals, max_val):
    '''Convert an int list of data into one-hot vectors.'''
    return np.eye(max_val)[np.array(hot_vals)]


def unwrap_env(env, until_class=None):
    '''Unwrap wrapped env until we get an instance that is a until_class.

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

def unwrap_to_subclass_of(env, superclass):
    '''
    Unwraps wrapped env until we get an instance that is a subclass of superclass.

    '''

    while hasattr(env, 'env') and not issubclass(type(env), superclass):
        env = env.env

    if not isinstance(env, superclass):
        raise ValueError(
            "Unwrapping env did not yield an instance which is subclass of  {}".format(
                superclass))
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

def is_unwrappable_to_subclass_of(env, superclass):
    '''Check if env can be unwrapped to to_class.'''

    if issubclass(type(env), superclass):
        return True
    while hasattr(env, 'env'):
        env = env.env
        if issubclass(type(env), superclass):
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

    discrete_env = unwrap_env(env, DiscreteEnv)

    if is_unwrappable_to(env, rew_wrapper.RewardWrapper):
        reward_wrapper = unwrap_env(env, rew_wrapper.RewardWrapper)
        reward_function = reward_wrapper.reward_function
        P_based_on_reward_function = {}
        for (state, P_for_state) in unwrap_env(env, DiscreteEnv).P.items():
            P_based_on_reward_function[state] = {}
            for (action, P_for_state_action) in P_for_state.items():
                outcomes = []
                for outcome in P_for_state_action:
                    P_entry = list(outcome)
                    next_state = outcome[1]
                    if is_unwrappable_to_subclass_of(env, feat_wrapper.FeatureWrapper):
                        reward_input = unwrap_to_subclass_of(env, feat_wrapper.FeatureWrapper).features(state, action, next_state)
                    else:
                        reward_input = reward_wrapper.get_reward_input_for(state, action, next_state)
                    P_entry[2] = reward_function.reward(reward_input).item()
                    outcomes.append(tuple(P_entry))
                P_based_on_reward_function[state][action] = outcomes
        discrete_env.P = P_based_on_reward_function
        print(P_based_on_reward_function)

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    rewards = np.zeros([n_states, n_actions])

    # Iterate over "from" states:
    for s, P_given_state in discrete_env.P.items():

        # Iterate over actions:
        for a, transitions in P_given_state.items():

            # Iterate over "to" states:
            for proba, sp, r, done in transitions:
                rewards[s, a] += r * proba

    return rewards


class MetricsLogger():
    '''Listens for metrics to be stored as json.

    Metrics can be stored once per run or once per training step. The
    simplest usage is to load the jsons of relevant runs, select
    metrics and convert to a pandas DataFrames.
    '''

    def __init__(self):
        self.metrics = defaultdict(lambda: [])

    def log_metric(self, name, value):
        self.metrics[name].append(value)

    def save(self, path):
        with open(path, 'wt') as f:
            json.dump(self.metrics, f)


def sigma(array):
    '''Replace negative entries of array w/ -1, others w/ 1.

    Returns modified array.
    '''
    array = array.copy()
    array[array >= 0] = 1
    array[array < 0] = -1
    return array
