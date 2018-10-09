import gym
import numpy as np

from irl_benchmark.rl.algorithms.tabular_q import TabularQ

'''Test Tabular Q-Learning on FrozenLake.'''

def test_frozen_finds_good_solution(duration=2):
    '''Check if goal was reached in at least 40% of 100 episodes.

    NOTE: For the default value of duration only checks if agent
    runs at all, not whether results fulfill above conditions.
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

    # Evaluate conditions on results only if duration was manually
    # set to be large.
    if duration < 5:
        return

    assert np.mean(episode_rewards) > 0.4
    assert np.max(episode_rewards) == 1.0
