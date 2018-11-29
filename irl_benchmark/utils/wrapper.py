"""Utils module containing wrapper specific helper functions."""

from typing import Type, Union

import gym


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
