import gym
import numpy as np
import pytest

from irl_benchmark.irl.feature import feature_wrapper
from irl_benchmark.irl.reward.reward_function import FeatureBasedRewardFunction
from irl_benchmark.irl.reward.reward_wrapper import RewardWrapper
from irl_benchmark.rl.algorithms.value_iteration import ValueIteration
from irl_benchmark.utils.utils import get_reward_matrix
'''Test Value Iteration on FrozenLake.'''


def test_frozen_finds_good_solution(duration=0.1):
    '''
    Check if goal was reached in at least 30% of 100 episodes.
    '''
    env = gym.make('FrozenLake-v0')
    agent = ValueIteration(env)
    agent.train(duration)
    N = 100
    episode_rewards = []
    for episode in range(N):
        episode_reward = 0
        state = env.reset()
        done = False
        while not done:
            state, reward, done, _ = env.step(agent.pick_action(state))
            episode_reward += reward
        episode_rewards.append(episode_reward)

    # Test saving / loading:
    fn = '/tmp/VI.pickle'
    agent.save(fn)
    agent2 = ValueIteration(env)
    agent2.load(fn)
    assert (agent.V == agent2.V).all()
    assert (agent.pi == agent2.pi).all()

    assert np.mean(episode_rewards) > 0.3
    assert np.max(episode_rewards) == 1.0


def test_true_reward_not_leaked(duration=0.1):
    env = feature_wrapper.make('FrozenLake-v0')
    reward_function = FeatureBasedRewardFunction(env, np.random.randn(16))
    env = RewardWrapper(env, reward_function)
    agent = ValueIteration(env)
    agent.train(0.5)
    assert np.all(
        agent.rewards == get_reward_matrix(env, with_absorbing_state=True))
    reward_function = FeatureBasedRewardFunction(env, np.random.randn(16))
    env.update_reward_function(reward_function)
    agent.train(0.1)
    assert np.all(
        agent.rewards == get_reward_matrix(env, with_absorbing_state=True))


def test_frozen_finds_good_solution_stoch(duration=0.1):
    '''
    Check if goal was reached in at least 30% of 100 episodes.
    '''
    env = gym.make('FrozenLake-v0')
    agent = ValueIteration(env, temperature=0.0002)
    agent.train(duration)
    N = 100
    episode_rewards = []
    for episode in range(N):
        episode_reward = 0
        state = env.reset()
        done = False
        while not done:
            state, reward, done, _ = env.step(agent.pick_action(state))
            episode_reward += reward
        episode_rewards.append(episode_reward)

    if duration >= 1.0:
        assert np.mean(episode_rewards) > 0.3
        assert np.max(episode_rewards) == 1.0


@pytest.mark.slow
def test_frozen_finds_good_solution_slow():
    test_frozen_finds_good_solution(1)


@pytest.mark.slow
def test_true_reward_not_leaked_slow():
    test_frozen_finds_good_solution(1)
