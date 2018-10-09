import gym
import numpy as np

from irl_benchmark.rl.algorithms.value_iteration import ValueIteration
'''Test Value Iteration on FrozenLake.'''


def test_frozen_finds_good_solution(duration=1):
    '''
    Check if goal was reached in at least 40% of 100 episodes.
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

    assert np.mean(episode_rewards) > 0.4
    assert np.max(episode_rewards) == 1.0
