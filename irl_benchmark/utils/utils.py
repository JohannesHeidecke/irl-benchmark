import numpy as np

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


def get_transition_matrix(env):
    """Gets transition matrix from discrete environment"""
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
    """Gets reward array from discrete environment"""
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
