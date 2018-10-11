import gym
import numpy as np
import pickle

from irl_benchmark.irl.algorithms.relent.relent import RelEnt
from irl_benchmark.irl.feature.feature_wrapper import FrozenLakeFeatureWrapper
from irl_benchmark.irl.feature.feature_wrapper import FeatureWrapper
from irl_benchmark.irl.collect import collect_trajs
from irl_benchmark.irl.reward.reward_function import FeatureBasedRewardFunction
from irl_benchmark.irl.reward.reward_wrapper import RewardWrapper
from irl_benchmark.rl.algorithms import TabularQ
from irl_benchmark.utils.utils import unwrap_env

# Define important script constants here:
store_to = 'data/frozen/expert/'
no_episodes = 1000
max_steps_per_episode = 100


def rl_alg_factory(env):
    '''Return an RL algorithm that will collect expert trajectories.'''
    return TabularQ(env)


# RelEnt IRL assumes that rewards are linear in features.
# However, FrozenLake doesn't provide features. It is sufficiently small
# to work with tabular methods. Therefore, we just use a wrapper that uses
# a one-hot encoding of the state space as features.
env = gym.make('FrozenLake-v0')
env = FrozenLakeFeatureWrapper(env)

# Generate expert trajectories.
expert_agent = rl_alg_factory(env)
print('Training expert agent...')
expert_agent.train(600)
print('Done training expert')
expert_trajs = collect_trajs(env, expert_agent, no_episodes,
                             max_steps_per_episode, store_to)

# you can comment out the previous block if expert data has already
# been generated and load the trajectories from file by uncommenting
# next 2 lines:
# with open(store_to + 'trajs.pkl', 'rb') as f:
#     expert_trajs = pickle.load(f)

# Provide random reward function as initial reward estimate.
# This probably isn't really required.
n_features = unwrap_env(env, FeatureWrapper).feature_shape()[0]
reward_function = FeatureBasedRewardFunction(env,
                                             np.random.normal(size=n_features))
env = RewardWrapper(env, reward_function)

# Run Relative Entropy IRL for by default 60 seconds.
relent = RelEnt(env, expert_trajs, rl_alg_factory, horizon=100, delta=.05)
relent.train(1e-3, 60, verbose=True)
