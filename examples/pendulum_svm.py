import gym
import numpy as np
import pickle

from irl_benchmark.irl.algorithms.appr.appr_irl import ApprIRL
from irl_benchmark.irl.feature.feature_wrapper import PendulumFeatureWrapper
from irl_benchmark.irl.collect import collect_trajs
from irl_benchmark.irl.reward.reward_function import FeatureBasedRewardFunction
from irl_benchmark.irl.reward.reward_wrapper import RewardWrapper
from irl_benchmark.rl.algorithms import PPO

# Define important script constants here:
store_to = 'data/pendulum/expert/'
no_episodes = 1000
max_steps_per_episode = 100

# Define the RL algorithm that is used both for the expert and
# in the IRL loop:
def rl_alg_factory(env):
    return PPO(env)

# Pendulum has features that can easily be extracted from the previous state, 
# see PendulumFeatureWrapper.
env = gym.make('Pendulum-v0')
env = PendulumFeatureWrapper(env)

# Generate expert trajectories.
expert_agent = rl_alg_factory(env)
print('Training expert agent...')
expert_agent.train(15)
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
reward_function = FeatureBasedRewardFunction(env, np.random.normal(size=16))
env = RewardWrapper(env, reward_function)

# Run projection algorithm for up to 5 minutes.
appr_irl = ApprIRL(env, expert_trajs, rl_alg_factory, proj=True)
appr_irl.train(600, verbose=True)
