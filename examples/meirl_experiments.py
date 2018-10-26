import gym
import numpy as np

from irl_benchmark.irl.algorithms.maxent.me_irl import MaxEnt
from irl_benchmark.irl.collect import collect_trajs
from irl_benchmark.irl.feature.feature_wrapper import FrozenLakeFeatureWrapper
from irl_benchmark.irl.reward.reward_function import FeatureBasedRewardFunction
from irl_benchmark.irl.reward.reward_wrapper import RewardWrapper
from irl_benchmark.rl.algorithms.value_iteration import ValueIteration
from irl_benchmark.utils.utils import get_transition_matrix

store_to = 'data/frozen/expert/'
no_episodes = 1000
max_steps_per_episode = 1000

env = gym.make('FrozenLake8x8-v0')
env = FrozenLakeFeatureWrapper(env)
initial_reward_function_estimate = FeatureBasedRewardFunction(
    env=env, parameters=np.zeros(64))
env = RewardWrapper(env=env, reward_function=initial_reward_function_estimate)

# Generate expert trajectories.
expert_agent = ValueIteration(env)
print('Training expert agent...')
expert_agent.train(30)
expert_trajs = collect_trajs(env, expert_agent, no_episodes,
                             max_steps_per_episode, store_to)

feat_map = np.eye(64)

transition_dynamics = get_transition_matrix(env)


def rl_alg_factory(env):
    return ValueIteration(env)


meirl = MaxEnt(
    env,
    expert_trajs=expert_trajs,
    transition_dynamics=transition_dynamics,
    rl_alg_factory=rl_alg_factory)
rewards = meirl.train(feat_map, time_limit=60, verbose=True)

rewards = (rewards - np.min(rewards)) / (np.max(rewards) - np.min(rewards))

print(rewards)
