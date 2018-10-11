import gym
from gym.envs.toy_text.discrete import DiscreteEnv

from irl_benchmark.irl.feature import feature_wrapper
from irl_benchmark.utils.utils import is_unwrappable_to

# All environments to be used need to be in the following list:
env_ids = ['FrozenLake-v0', 'FrozenLake8x8-v0', 'Pendulum-v0']

def envs_feature_based():
    '''Return all environments for which a feature wrapper exists.'''
    return set(feature_wrapper.feature_wrappable_envs())

def envs_known_transitions():
    '''Return all environments for which transition dynamics are known.'''
    result = []
    for env_id in env_ids:
        if is_unwrappable_to(gym.make(env_id),DiscreteEnv):
            result.append(env_id)
    return set(result)

