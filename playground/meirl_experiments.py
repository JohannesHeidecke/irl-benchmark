import numpy as np

import gym

from irl_benchmark.irl.algorithms.maxent.meirl import MaxEnt
from irl_benchmark.irl.collect import collect_trajs
from irl_benchmark.irl.feature.feature_wrapper import FrozenLakeFeatureWrapper
from irl_benchmark.rl.algorithms import TabularQ
from irl_benchmark.utils.utils import get_transition_matrix

store_to = 'data/frozen/expert/'
no_episodes = 1000
max_steps_per_episode = 100

env = gym.make('FrozenLake-v0')
env = FrozenLakeFeatureWrapper(env)

# Generate expert trajectories.
expert_agent = TabularQ(env)
print('Training expert agent...')
expert_agent.train(1)
print('Done training expert')
expert_trajs = collect_trajs(env, expert_agent, no_episodes,
                             max_steps_per_episode, store_to)

feat_map = np.eye(16)

transition_dynamics = get_transition_matrix(env)


def rl_alg_factory(env):
    return TabularQ(env)


meirl = MaxEnt(env, transition_dynamics=transition_dynamics, rl_alg_factory=rl_alg_factory, expert_trajs=expert_trajs)
rewards = meirl.train(feat_map)

print(rewards)
