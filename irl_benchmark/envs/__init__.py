"""Module dealing with creating and administration of environments."""
from typing import Callable

import gym
from gym.envs.toy_text.discrete import DiscreteEnv
from gym.wrappers import TimeLimit

from irl_benchmark.envs.maze_world import MazeWorld
from irl_benchmark.irl.feature import feature_wrapper
from irl_benchmark.irl.reward.reward_function import BaseRewardFunction
from irl_benchmark.irl.reward.reward_wrapper import RewardWrapper
from irl_benchmark.rl.model.discrete_env import DiscreteEnvModelWrapper
from irl_benchmark.rl.model.maze_world import MazeModelWrapper
import irl_benchmark.utils as utils

# All environments to be used need to be in the following list:
ENV_IDS = [
    'FrozenLake-v0', 'FrozenLake8x8-v0', 'MazeWorld0-v0', 'MazeWorld1-v0'
]

ENV_IDS_NON_GYM = ['MazeWorld0-v0', 'MazeWorld1-v0']

ENV_IDS_ACTION_IN_DOMAIN = ['MazeWorld0-v0', 'MazeWorld1-v0']
ENV_IDS_NEXT_STATE_IN_DOMAIN = []


def make_wrapped_env(env_id: str,
                     with_feature_wrapper: bool = False,
                     reward_function_factory: Callable = None,
                     with_model_wrapper: bool = False):
    """Make an environment, potentially wrapped in FeatureWrapper, RewardWrapper,
    and BaseWorldModelWrapper.

    Parameters
    ----------
    env_id: str
        The environment's id, e.g. 'FrozenLake-v0'.
    with_feature_wrapper: bool
        Whether to use a feature wrapper.
    reward_function_factory: Callable
        A function which returns a new reward function when called. If this is
        provided, the environment will be wrapped in a RewardWrapper using
        the returned reward function.
    with_model_wrapper: bool
        Whether to use a BaseWorldModelWrapper.

    Returns
    -------
    gym.Env
        A gym environment, potentially wrapped.
    """
    assert env_id in ENV_IDS
    if with_feature_wrapper:
        assert env_id in feature_wrapper.feature_wrappable_envs()
        env = feature_wrapper.make(env_id)
    else:
        env = make_env(env_id)

    if reward_function_factory is not None:
        reward_function = reward_function_factory(env)
        assert isinstance(reward_function, BaseRewardFunction)
        env = RewardWrapper(env, reward_function)

    if with_model_wrapper:
        if utils.wrapper.is_unwrappable_to(env, DiscreteEnv):
            env = DiscreteEnvModelWrapper(env)
        elif utils.wrapper.is_unwrappable_to(env, MazeWorld):
            env = MazeModelWrapper(env)
        else:
            raise NotImplementedError()
    return env


def make_env(env_id: str):
    """Make a basic gym environment, without any special wrappers.

    Parameters
    ----------
    env_id: str
        The environment's id, e.g. 'FrozenLake-v0'.
    Returns
    -------
    gym.Env
        A gym environment.
    """
    assert env_id in ENV_IDS
    if not env_id in ENV_IDS_NON_GYM:
        env = gym.make(env_id)
    else:
        if env_id == 'MazeWorld0-v0':
            env = TimeLimit(MazeWorld(map_id=0), max_episode_steps=200)
        elif env_id == 'MazeWorld1-v0':
            env = TimeLimit(MazeWorld(map_id=1), max_episode_steps=200)
        else:
            raise NotImplementedError()
    return env


def envs_feature_based():
    """Return all environment ids for which a feature wrapper exists."""
    return set(feature_wrapper.feature_wrappable_envs())


def envs_known_transitions():
    """Return all environment ids for which transition dynamics are known."""
    result = []
    for env_id in ENV_IDS:
        if utils.wrapper.is_unwrappable_to(make_env(env_id), DiscreteEnv):
            result.append(env_id)
    return set(result)
