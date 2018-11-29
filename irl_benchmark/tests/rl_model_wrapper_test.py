import numpy as np
import sparse

from irl_benchmark.envs import make_env, make_wrapped_env
from irl_benchmark.irl.reward.reward_function import TabularRewardFunction, FeatureBasedRewardFunction
from irl_benchmark.rl.model.discrete_env import DiscreteEnvModelWrapper


def test_frozenlake_transitions():
    env = make_env('FrozenLake-v0')
    env = DiscreteEnvModelWrapper(env)
    transitions = env.get_transition_array()

    # assert probability sums to 1.0
    for s in range(transitions.shape[0]):
        for a in range(transitions.shape[1]):
            assert transitions[s, a].sum() == 1.0

    assert isinstance(transitions, np.ndarray)
    assert transitions.shape == (17, 4, 17)

    # check state distribution under random policy:
    S = np.zeros(17)
    S[0] = 1.
    transitions_random_pol = np.sum(transitions, axis=1) * 0.25

    S_1 = S.dot(np.linalg.matrix_power(transitions_random_pol, 1))
    S_2 = S.dot(np.linalg.matrix_power(transitions_random_pol, 2))
    S_50 = S.dot(np.linalg.matrix_power(transitions_random_pol, 50))
    assert S_1[-1] == 0
    assert S_2[-1] == 0
    assert S_50[-1] > .999
    for i in range(50):
        S_i = S.dot(np.linalg.matrix_power(transitions_random_pol, i))
        assert np.isclose(np.sum(S_i), 1.)


def test_frozenlake_wrapped_transitions():
    env = make_wrapped_env('FrozenLake-v0', with_feature_wrapper=True)
    env = DiscreteEnvModelWrapper(env)
    transitions = env.get_transition_array()

    # assert probability sums to 1.0
    for s in range(transitions.shape[0]):
        for a in range(transitions.shape[1]):
            assert transitions[s, a].sum() == 1.0

    assert isinstance(transitions, np.ndarray)
    assert transitions.shape == (17, 4, 17)


def test_frozenlake_fully_wrapped_transitions():
    env = make_wrapped_env(
        'FrozenLake-v0', with_feature_wrapper=True, with_model_wrapper=True)
    transitions = env.get_transition_array()

    # assert probability sums to 1.0
    for s in range(transitions.shape[0]):
        for a in range(transitions.shape[1]):
            assert transitions[s, a].sum() == 1.0

    assert isinstance(transitions, np.ndarray)
    assert transitions.shape == (17, 4, 17)


def test_frozenlake8_transitions():
    env = make_env('FrozenLake8x8-v0')
    env = DiscreteEnvModelWrapper(env)
    transitions = env.get_transition_array()

    # assert probability sums to 1.0
    for s in range(transitions.shape[0]):
        for a in range(transitions.shape[1]):
            assert transitions[s, a].sum() == 1.0

    assert isinstance(transitions, np.ndarray)
    assert transitions.shape == (65, 4, 65)


def test_frozenlake_rewards():
    env = make_env('FrozenLake-v0')
    env = DiscreteEnvModelWrapper(env)
    transitions = env.get_transition_array()
    rewards = env.get_reward_array()

    assert rewards.shape == (17, 4)
    assert transitions.shape == (17, 4, 17)

    true_rews = np.zeros(16 + 1)
    # [-2] since [-1] is the added absorbing state
    true_rews[-2] = 1.0

    for s in range(16 + 1):
        for a in range(4):
            assert np.isclose(rewards[s, a],
                              transitions[s, a, :].dot(true_rews))


def test_frozenlake8x8_rewards():
    env = make_env('FrozenLake8x8-v0')
    env = DiscreteEnvModelWrapper(env)
    transitions = env.get_transition_array()
    rewards = env.get_reward_array()

    assert rewards.shape == (65, 4)
    assert transitions.shape == (65, 4, 65)

    true_rews = np.zeros(64 + 1)
    # [-2] since [-1] is the added absorbing state
    true_rews[-2] = 1.0

    for s in range(64 + 1):
        for a in range(4):
            assert np.isclose(rewards[s, a],
                              transitions[s, a, :].dot(true_rews))


def test_frozenlake8x8_rewards_tabular():

    true_rews = np.random.randn(65)
    true_rews[-1] = 0

    def reward_function_factory(env):
        return TabularRewardFunction(env, true_rews[:-1])

    env = make_wrapped_env(
        'FrozenLake8x8-v0',
        with_feature_wrapper=True,
        reward_function_factory=reward_function_factory,
        with_model_wrapper=True)

    transitions = env.get_transition_array()
    rewards = env.get_reward_array()

    assert rewards.shape == (65, 4)
    assert transitions.shape == (65, 4, 65)

    for s in range(64 + 1):
        for a in range(4):
            assert np.isclose(rewards[s, a],
                              transitions[s, a, :].dot(true_rews))


def test_get_reward_matrix_wrapped_feature():

    true_rews = np.random.randn(65)
    true_rews[-1] = 0

    def reward_function_factory(env):
        return FeatureBasedRewardFunction(env, true_rews[:-1])

    env = make_wrapped_env(
        'FrozenLake8x8-v0',
        with_feature_wrapper=True,
        reward_function_factory=reward_function_factory,
        with_model_wrapper=True)

    transitions = env.get_transition_array()
    rewards = env.get_reward_array()

    assert rewards.shape == (65, 4)
    assert transitions.shape == (65, 4, 65)

    for s in range(64 + 1):
        for a in range(4):
            assert np.isclose(rewards[s, a],
                              transitions[s, a, :].dot(true_rews))


def test_maze_indices():
    env = make_wrapped_env('MazeWorld0-v0', with_model_wrapper=True)
    for i in range(10240):
        assert i == env.state_to_index(env.index_to_state(i))