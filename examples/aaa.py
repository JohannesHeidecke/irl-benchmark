import gym
import numpy as np

from irl_benchmark.irl.algorithms.appr_irl import ApprIRL
from irl_benchmark.irl.collect import collect_trajs
from irl_benchmark.irl.feature import feature_wrapper
from irl_benchmark.irl.reward.reward_function import FeatureBasedRewardFunction
from irl_benchmark.irl.reward.reward_wrapper import RewardWrapper
from irl_benchmark.rl.algorithms.value_iteration import ValueIteration


# TODO: this is an example for Sayan, delete later.

store_to = 'data/frozen/expert/'
no_episodes = 500


def rl_alg_factory(env):
    return ValueIteration(env, {'gamma': 0.9})


env = feature_wrapper.make('FrozenLake-v0')

expert_agent = rl_alg_factory(env)
expert_agent.train(15)
expert_trajs = collect_trajs(env, expert_agent, no_episodes, None, store_to)

# wrap env in random reward function to prevent leaking true reward:
reward_function = FeatureBasedRewardFunction(env, np.random.normal(size=16))
env = RewardWrapper(env, reward_function)

# Run projection algorithm for up to 5 minutes.
irl_config = {'gamma': 0.9, 'verbose': True}
appr_irl = ApprIRL(env, expert_trajs, rl_alg_factory, irl_config)
reward_function, rl_agent = appr_irl.train(
    no_irl_iterations=50,
    no_rl_episodes_per_irl_iteration=no_episodes,
    no_irl_episodes_per_irl_iteration=no_episodes)

print(reward_function.parameters)
