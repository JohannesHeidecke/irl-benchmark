"""Utils module containing wrapper specific helper functions."""

from copy import copy
from typing import Type, Union

import gym
from gym.envs.toy_text.discrete import DiscreteEnv
import numpy as np

import irl_benchmark.irl.feature.feature_wrapper as feature_wrapper
from irl_benchmark.irl.reward.reward_wrapper import RewardWrapper
import irl_benchmark.irl.reward.reward_function as rew_funcs


def unwrap_env(env: gym.Env,
               until_class: Union[None, gym.Env] = None) -> gym.Env:
    """Unwrap wrapped env until we get an instance that is a until_class.

    If until_class is None, env will be unwrapped until the lowest layer.
    """
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


def is_unwrappable_to(env: gym.Env, to_wrapper: Type[gym.Wrapper]) -> bool:
    """Check if env can be unwrapped to to_wrapper.

    Parameters
    ----------
    env: gym.Env
        A gym environment (potentially wrapped).
    to_wrapper: Type[gym.Wrapper]
        A wrapper class extending gym.Wrapper.

    Returns
    -------
    bool
        True if env could be unwrapped to desired wrapper, False otherwise.
    """
    if isinstance(env, to_wrapper):
        return True
    while hasattr(env, 'env'):
        env = env.env
        if isinstance(env, to_wrapper):
            return True
    return False


def get_transition_matrix(env: gym.Env, with_absorbing_state: bool = True):
    """Get transition matrix from discrete environment.

    Parameters
    ----------
    env: gym.Env
        A gym environment, unwrappable to DiscreteEnv.
    with_absorbing_state: bool
        Use an absorbing state which is reached whenever a step returns done.

    Returns
    -------
    np.ndarray
        An array containing transition probabilities of shape (n_states, n_actions, n_states)
        or (n_states + 1, n_actions, n_states + 1) if an absorbing state is used.
    """
    env = unwrap_env(env, DiscreteEnv)

    n_states = env.observation_space.n
    if with_absorbing_state:
        # adding one state in the end of the table as absorbing state:
        n_states += 1
    n_actions = env.action_space.n

    table = np.zeros([n_states, n_actions, n_states])

    # Iterate over "from" states:
    for state, transitions_given_state in env.P.items():

        # Iterate over actions:
        for action, transitions in transitions_given_state.items():

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


def get_reward_matrix(env: gym.Env, with_absorbing_state: bool = True):
    """Get reward array from discrete environment.

    If env is wrapped in a RewardWrapper, use the wrapper's rewards instead of
    the original ones.

    Parameters
    ----------
    env: gym.Env
        A gym environment, unwrappable to DiscreteEnv.
    with_absorbing_state: bool
        Use an absorbing state which is reached whenever a step returns done.

    Returns
    -------
    np.ndarray
        A numpy array of shape (n_states, n_actions) or (n_states + 1, n_actions) if
        with_absorbing_state == True. Contains values of R(s, a)
    """

    n_states = env.observation_space.n
    if with_absorbing_state:
        n_states += 1
    n_actions = env.action_space.n

    # unwrap the discrete env from which transitions can be extracted:
    discrete_env = unwrap_env(env, DiscreteEnv)
    # by default this discrete env's variable P will be used to extract rewards
    correct_transitions = copy(discrete_env.P)

    if with_absorbing_state:
        # change transitions in a way that terminal states map to the absorbing state
        for (state, transitions_for_state) in correct_transitions.items():
            for (action, transitions_for_state_action
                 ) in transitions_for_state.items():
                if len(transitions_for_state_action) == 1 \
                        and transitions_for_state_action[0][1] == state \
                        and transitions_for_state_action[0][3] is True:
                    # state maps to itself, but should map to absorbing state
                    rewired_outcome = (1.0, n_states - 1, 0, True)
                    correct_transitions[state][action] = [rewired_outcome]

    # however, if there is a reward wrapper, we need to use the
    # wrapped reward function:
    if is_unwrappable_to(env, RewardWrapper):
        # get the reward function:
        reward_wrapper = unwrap_env(env, RewardWrapper)
        reward_function = reward_wrapper.reward_function
        # re-calculate transitions based on reward function:
        transitions_based_on_reward_function = {}
        for (state, transitions_for_state) in correct_transitions.items():
            transitions_based_on_reward_function[state] = {}
            for (action, transitions_for_state_action
                 ) in transitions_for_state.items():
                outcomes = []
                for old_outcome in transitions_for_state_action:
                    outcome = list(copy(old_outcome))
                    next_state = outcome[1]
                    if with_absorbing_state and next_state == n_states - 1:
                        # hard coded: 0 reward when going to absorbing state
                        # (since the absorbing state is added artificially and
                        # not part of the wrapper's reward function)
                        outcome[2] = 0.0
                    else:
                        if isinstance(reward_function,
                                      rew_funcs.FeatureBasedRewardFunction):
                            # reward function needs features as input
                            reward_input = unwrap_env(
                                env, feature_wrapper.FeatureWrapper).features(
                                    None, None, next_state)
                        elif isinstance(reward_function,
                                        rew_funcs.TabularRewardFunction):
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
                transitions_based_on_reward_function[state][action] = outcomes
        correct_transitions = transitions_based_on_reward_function

    rewards = np.zeros([n_states, n_actions])

    # Iterate over "from" states:
    for state, transitions_given_state in correct_transitions.items():

        # Iterate over actions:
        for action, transitions in transitions_given_state.items():

            # Iterate over "to" states:
            for probability, _, reward, _ in transitions:
                rewards[state, action] += reward * probability
    if with_absorbing_state:
        rewards[-1, :] = 0.0
    return rewards
