import json
from collections import defaultdict
from copy import copy
import matplotlib.pyplot as pl
import numpy as np
from gym.envs.toy_text.discrete import DiscreteEnv

from irl_benchmark.irl.reward.reward_function import FeatureBasedRewardFunction, TabularRewardFunction
import irl_benchmark.irl.reward.reward_wrapper as rew_wrapper
import irl_benchmark.irl.feature.feature_wrapper as feat_wrapper


def to_one_hot(hot_vals, max_val, zeros_fun=np.zeros):
    '''Convert an int, or a list of ints, to a one-hot array of floats.

    `zeros_fun` controls which function is used to create the array. It should
    be either `numpy.zeros` or `torch.zeros`.
    '''
    try:
        N = len(hot_vals)
        res = zeros_fun((N, max_val))
        res[np.arange(N), hot_vals] = 1.
    except TypeError:
        res = zeros_fun((max_val,))
        res[hot_vals] = 1.
    return res


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


def is_unwrappable_to(env, to_class):
    '''Check if env can be unwrapped to to_class.'''
    if isinstance(env, to_class):
        return True
    while hasattr(env, 'env'):
        env = env.env
        if isinstance(env, to_class):
            return True
    return False


def get_transition_matrix(env, with_absorbing_state=True):
    '''Gets transition matrix from discrete environment.'''

    env = unwrap_env(env, DiscreteEnv)

    n_states = env.observation_space.n
    if with_absorbing_state:
        # adding one state in the end of the table as absorbing state:
        n_states += 1
    n_actions = env.action_space.n

    table = np.zeros([n_states, n_actions, n_states])

    # Iterate over "from" states:
    for state, P_given_state in env.P.items():

        # Iterate over actions:
        for action, transitions in P_given_state.items():

            # Iterate over "to" states:
            for probability, next_state, _, done in transitions:
                table[state, action, next_state] += probability
                if done and with_absorbing_state is True:
                    # map next_state to absorbing state:
                    table[next_state, :, :] = 0.0
                    table[next_state, :, -1] = 1.0

    if with_absorbing_state:
        # absorbing state that is reached whenever done == True
        # only reaches itself for each action
        table[-1, :, -1] = 1.0

    return table


def get_reward_matrix(env, with_absorbing_state=True):
    '''Gets reward array from discrete environment.'''

    n_states = env.observation_space.n
    if with_absorbing_state:
        n_states += 1
    n_actions = env.action_space.n

    # unwrap the discrete env from which transitions can be extracted:
    discrete_env = unwrap_env(env, DiscreteEnv)
    # by default this discrete env's variable P will be used to extract rewards
    correct_P = copy(discrete_env.P)

    if with_absorbing_state:
        # change P in a way that terminal states map to the absorbing state
        for (state, P_for_state) in correct_P.items():
            for (action, P_for_state_action) in P_for_state.items():
                if len(P_for_state_action) == 1 and P_for_state_action[0][1] == state \
                        and P_for_state_action[0][3] is True:
                    # state maps to itself, but should map to absorbing state
                    rewired_outcome = (1.0, n_states - 1, 0, True)
                    correct_P[state][action] = [rewired_outcome]

    # however, if there is a reward wrapper, we need to use the
    # wrapped reward function:
    if is_unwrappable_to(env, rew_wrapper.RewardWrapper):
        # get the reward function:
        reward_wrapper = unwrap_env(env, rew_wrapper.RewardWrapper)
        reward_function = reward_wrapper.reward_function
        # re-calculate P based on reward function:
        P_based_on_reward_function = {}
        for (state, P_for_state) in correct_P.items():
            P_based_on_reward_function[state] = {}
            for (action, P_for_state_action) in P_for_state.items():
                outcomes = []
                for old_outcome in P_for_state_action:
                    outcome = list(copy(old_outcome))
                    next_state = outcome[1]
                    if with_absorbing_state and next_state == n_states - 1:
                        # hard coded: 0 reward when going to absorbing state
                        # (since the absorbing state is added artificially and
                        # not part of the wrapper's reward function)
                        outcome[2] = 0.0
                    else:
                        if isinstance(reward_function,
                                      FeatureBasedRewardFunction):
                            # reward function needs features as input
                            reward_input = unwrap_env(
                                env, feat_wrapper.FeatureWrapper).features(
                                    None, None, next_state)
                        elif isinstance(reward_function,
                                        TabularRewardFunction):
                            # reward function needs domain batch as input
                            assert reward_function.action_in_domain is False
                            assert reward_function.next_state_in_domain is False
                            reward_input = reward_wrapper.get_reward_input_for(
                                None, None, next_state)
                        else:
                            raise ValueError(
                                'The RewardWrapper\'s reward_function is' +
                                'of not supported type ' +
                                str(type(reward_function)))
                        # update the reward part of the outcome
                        outcome[2] = reward_function.reward(
                            reward_input).item()
                    outcomes.append(tuple(outcome))
                P_based_on_reward_function[state][action] = outcomes
        correct_P = P_based_on_reward_function

    rewards = np.zeros([n_states, n_actions])

    # Iterate over "from" states:
    for s, P_given_state in correct_P.items():

        # Iterate over actions:
        for a, transitions in P_given_state.items():

            # Iterate over "to" states:
            for proba, sp, r, done in transitions:
                rewards[s, a] += r * proba
    if with_absorbing_state:
        rewards[-1, :] = 0.0
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


def avg_undiscounted_return(trajs):
    '''Return average undiscounted true return of trajs.

    Args:
    trajs -- `list` of dictionaries w/ keys 'true_rewards' and 'rewards'
    '''
    total_true_reward = 0
    if len(trajs[0]['true_rewards']) > 0:
        for traj in trajs:
            total_true_reward += np.sum(traj['true_rewards'])
    else:
        for traj in trajs:
            total_true_reward += np.sum(traj['rewards'])
    avg_undiscounted_return = total_true_reward / len(trajs)
    return avg_undiscounted_return


def plot(data):
    '''Make shiny plots for the presentation on Saturday.
    Args:
    data -- `np.ndarray` of default shape (2, 5, 4, 3, 50)
    n_envs -- `int`, number of environments in the data
    n_algs -- `int`, number of IRL algorithms in the data
    n_exp_trajs -- `int`, number of numbers of expert trajectories in the data.
                   E.g. if an alg was run once w/ each of 10, 100, and 100
                   expert trajectories then n_exp_trajs would be 3
    n_metrics -- `int`, number of metrics in the data
    n_seeds -- `int`, number of independent runs in the data
    '''
    pl.rcParams['font.size'] = 16

    # Create fake data for testing.
    if data is None:
        data = {
            'environment_labels': ['FrozenLake', 'FrozenLake8x8'],
            'algorithm_labels': ['Appr-SVM', 'Appr-Proj',
                                 'MaxEnt', 'MaxCausalEnt', 'RelEnt'],
            'metric_labels': ['L2 error', 'true return',
                              'inverse learning loss'],
            'n_trajs_list': [10 ** n for n in range(1, 6)],
            'results': np.random.randn(2, 5, 3, 5, 50),
            }

    # Unpack data dictionary.
    envs = data['environment_labels']
    algs = data['algorithm_labels']
    metrics = data['metric_labels']
    n_trajs_list = data['n_trajs_list']
    results = data['results']
    n_envs = len(envs)
    n_algs = len(algs)
    n_metrics = len(metrics)
    n_trajs_list = np.array(n_trajs_list)
    data_std = np.std(results, axis=4)
    data_mean = np.mean(results, axis=4)
    data = np.stack([data_mean - data_std, data_mean, data_mean +
                     data_std], axis=4)

    # fig = pl.figure(figsize=[300, 300])
    fig, ax_lst = pl.subplots(n_algs, n_envs, sharex=True, figsize=[20, 13])
    fig.suptitle('Inverse Reinforcement Learning Benchmark')
    for env in range(n_envs):
        for alg in range(n_algs):
            for metric in range(n_metrics):
                ax = ax_lst[alg, env]
                ax.plot(n_trajs_list, data[env, alg, metric, :, 1],
                        label=metrics[metric])
                ax.fill_between(n_trajs_list,
                                data[env, alg, metric, :, 0],
                                data[env, alg, metric, :, 2],
                                alpha=.5)
                if env == 0:
                    ax.set_ylabel(algs[alg])
                if env == 0 and alg == 0:
                    ax.legend(loc='upper center', bbox_to_anchor=(1.0, 1.6),
                              ncol=3, fancybox=True, shadow=True)
                if alg == 0:
                    ax.set_title(envs[env])
                if alg == n_algs - 1:
                    ax.set_xlabel('Number of expert trajectories')
    pl.xscale('log')
    pl.savefig('foo.pdf')
    return None
