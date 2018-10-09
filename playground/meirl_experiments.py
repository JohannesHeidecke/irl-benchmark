import pickle

import gym

from irl_benchmark.irl.algorithms.maxent.meirl import MaxEnt
from irl_benchmark.irl.collect import collect_trajs
from irl_benchmark.irl.feature.feature_wrapper import FrozenLakeFeatureWrapper
from irl_benchmark.rl.algorithms import TabularQ


store_to = 'data/frozen/expert/'
no_episodes = 1000
max_steps_per_episode = 100

env = gym.make('FrozenLake-v0')
env = FrozenLakeFeatureWrapper(env)

# Generate expert trajectories.
expert_agent = TabularQ(env)
print('Training expert agent...')
expert_agent.train(15)
print('Done training expert')
expert_trajs = collect_trajs(env, expert_agent, no_episodes,
                             max_steps_per_episode, store_to)


meirl = MaxEnt(env, expert_trajs, )
rewards = meirl.train()

print(rewards)

