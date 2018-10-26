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
from irl_benchmark.rl.algorithms.value_iteration import ValueIteration
from irl_benchmark.rl.algorithms.ppo import PPO
from irl_benchmark.utils.utils import unwrap_env, avg_undiscounted_return

# Define important script constants here:
store_to = 'data/frozen/expert/'
no_episodes = 1000
max_steps_per_episode = 500


def rl_alg_factory(env, lp=False):
    '''Return an RL algorithm that will collect expert trajectories.'''
    if lp:
        return ValueIteration(env, error=1e-5)
    else:
        return TabularQ(env)


# RelEnt IRL assumes that rewards are linear in features.
# However, FrozenLake doesn't provide features. It is sufficiently small
# to work with tabular methods. Therefore, we just use a wrapper that uses
# a one-hot encoding of the state space as features.
env = gym.make('FrozenLake-v0')
env = FrozenLakeFeatureWrapper(env)

# Generate expert trajectories.
expert_agent = rl_alg_factory(env, lp=True)
print('Training expert agent...')
expert_agent.train(600)
print('Done training expert')
expert_trajs = collect_trajs(env, expert_agent, no_episodes,
                             max_steps_per_episode, store_to)
expert_performance = avg_undiscounted_return(expert_trajs)
print('The expert ' + 'reached the goal in ' + str(expert_performance) +
      ' of trajs.')

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
relent = RelEnt(env, expert_trajs, rl_alg_factory, horizon=500, delta=.01)
relent.train(1e-3, 60, verbose=False)

# Check how often the goal state was reached.
test_agent = rl_alg_factory(relent.env, lp=True)
print('Training test agent w/ reward obtained by RelEnt...')
test_agent.train(60)
print('Done training test agent.')
test_trajs = collect_trajs(
    relent.env,
    test_agent,
    no_episodes,
    max_steps_per_episode,
    store_to='data/frozen/foo')
irl_performance = avg_undiscounted_return(test_trajs)
print('The test agent trained w/ the reward estimated by RelEnt IRL ' +
      'reached the goal in ' + str(irl_performance) + ' of trajs.')
