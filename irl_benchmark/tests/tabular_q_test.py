import gym
import numpy as np
import pytest

from irl_benchmark.rl.algorithms.tabular_q import TabularQ
'''Test Tabular Q-Learning on FrozenLake.'''


def test_frozen_finds_good_solution(duration=0.1):
    '''
    Check if goal was reached in at least 40% of 100 episodes.
    '''
    env = gym.make('FrozenLake-v0')
    agent = TabularQ(env)
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
    fn = '/tmp/q_agent.pickle'
    agent.save(fn)
    agent2 = TabularQ(env)
    agent2.load(fn)
    for k, v in agent.Q.items():
        assert (v == agent2.Q[k]).all()

    if duration >= 1:
        assert np.mean(episode_rewards) > 0.3
        assert np.max(episode_rewards) == 1.0


@pytest.mark.slow
def test_frozen_finds_good_solution_slow():
    test_frozen_finds_good_solution(2)
