import gym
from gym.envs.toy_text.discrete import DiscreteEnv

from irl_benchmark.utils.utils import unwrap_env, get_transition_matrix
from irl_benchmark.irl.feature.feature_wrapper import FrozenLakeFeatureWrapper


def test_unwrap():
    env = gym.make('FrozenLake-v0')
    assert env.env is unwrap_env(env, DiscreteEnv)

    # No unwrapping needed:
    assert env is unwrap_env(env, gym.Env)

    # Unwrap all the way:
    assert env.env is unwrap_env(env)

    env = FrozenLakeFeatureWrapper(env)
    assert env.env.env is unwrap_env(env, DiscreteEnv)

    # No unwrapping needed:
    assert env is unwrap_env(env, FrozenLakeFeatureWrapper)

    # Unwrap all the way:
    assert env.env.env is unwrap_env(env)


def test_get_transition_matrix():
    env = gym.make('FrozenLake-v0')
    table = get_transition_matrix(env)

    # Assert probability sums to 1.0 (or zero if impossible to escape)
    for s in range(table.shape[0]):
        for a in range(table.shape[1]):
            assert table[s, a].sum() == 1.0 or table[s, a].sum() == 0.0

    env = FrozenLakeFeatureWrapper(env)
    table = get_transition_matrix(env)

    # Assert probability sums to 1.0 (or zero if impossible to escape)
    for s in range(table.shape[0]):
        for a in range(table.shape[1]):
            assert table[s, a].sum() == 1.0 or table[s, a].sum() == 0.0
