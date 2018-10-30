import gym
import numpy as np

from irl_benchmark.irl.feature import feature_wrapper
from irl_benchmark.irl.reward.reward_function import FeatureBasedRewardFunction
from irl_benchmark.irl.reward.reward_wrapper import RewardWrapper
from irl_benchmark.rl.algorithms import ValueIteration


def test_value_iteration():
    # gamma = 1.0
    env = gym.make('FrozenLake-v0')
    agent = ValueIteration(env, {'gamma': 1.0})
    agent.train(10)
    state_values = agent.state_values
    assert isinstance(state_values, np.ndarray)
    assert state_values.shape == (17, )
    # argmax should be state just before frisbee
    # (15 is final state, 16 is absorbing state)
    assert np.argmax(state_values) == 14
    assert state_values[14] > 0.93 and state_values[14] < 0.95
    assert state_values[15] == 0

    # gamma = 0.9
    env = gym.make('FrozenLake-v0')
    agent = ValueIteration(env, {'gamma': 0.9})
    agent.train(10)
    state_values = agent.state_values
    assert isinstance(state_values, np.ndarray)
    assert state_values.shape == (17, )
    # argmax should be state just before frisbee
    # (15 is final state, 16 is absorbing state)
    assert np.argmax(state_values) == 14
    assert state_values[14] > 0.63 and state_values[14] < 0.65
    # holes and frisbee should have zero value:
    for i in [5, 7, 11, 12, 15]:
        assert state_values[i] == 0

    # check some q values:
    # go right in second to last state
    assert np.argmax(agent.q_values[14, :]) == 1
    assert np.min(agent.q_values) == 0
    assert np.max(agent.q_values) <= 1

    # check policy:
    for i in range(16):
        assert np.isclose(np.sum(agent.policy(i)), 1.)
        assert np.min(agent.policy(i)) >= 0.
        assert np.argmax(agent.q_values[i, :]) == np.argmax(agent.policy(i))

    # check softmax policy
    old_state_values = agent.state_values
    old_q_values = agent.q_values
    agent = ValueIteration(env, {'gamma': 0.9, 'temperature': 0.1})
    agent.train(10)
    assert np.all(agent.state_values <= old_state_values)
    # at least initial state should now have lower value:
    assert agent.state_values[0] < old_state_values[0]
    assert np.all(agent.q_values <= old_q_values)
    # check policy:
    for i in range(16):
        assert np.isclose(np.sum(agent.policy(i)), 1.)
        assert np.min(agent.policy(i)) >= 0.
        assert np.argmax(agent.q_values[i, :]) == np.argmax(agent.policy(i))
        # ordering of probabilities should stay the same with softmax
        assert np.all(
            np.argsort(old_q_values[i, :]) == np.argsort(agent.policy(i)))

    # test policy array:
    policy_array = agent.policy_array()
    assert policy_array.shape == (17, 4)
    for i in range(16):
        assert np.all(agent.policy(i) == policy_array[i, :])

    # check if true reward isn't leaked:
    env = feature_wrapper.make('FrozenLake-v0')
    reward_function = FeatureBasedRewardFunction(env, np.zeros(16))
    env = RewardWrapper(env, reward_function)
    agent = ValueIteration(env, {})
    agent.train(10)
    assert np.sum(agent.state_values == 0)
