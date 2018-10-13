import gym
import numpy as np

from irl_benchmark.irl.algorithms.maxent.me_irl import MaxEnt
from irl_benchmark.irl.collect import collect_trajs
from irl_benchmark.irl.feature.feature_wrapper import FrozenLakeFeatureWrapper
from irl_benchmark.irl.reward.reward_function import FeatureBasedRewardFunction
from irl_benchmark.irl.reward.reward_wrapper import RewardWrapper
from irl_benchmark.rl.algorithms.value_iteration import ValueIteration
from irl_benchmark.utils.utils import get_transition_matrix, avg_undiscounted_return

store_to = 'data/frozen/expert/'
no_episodes = 1000
max_steps_per_episode = 500

env = gym.make('FrozenLake-v0')
env = FrozenLakeFeatureWrapper(env)
# Generate expert trajectories.
expert_agent = ValueIteration(env)
print('Training expert agent...')
expert_agent.train(30)
expert_trajs = collect_trajs(env, expert_agent, no_episodes,
                             max_steps_per_episode, store_to)
initial_reward_function_estimate = FeatureBasedRewardFunction(env=env, parameters=np.zeros(16))
env = RewardWrapper(env=env, reward_function=initial_reward_function_estimate)

feat_map = np.eye(16)

transition_dynamics = get_transition_matrix(env)


def rl_alg_factory(env):
    return ValueIteration(env)


expert_performance = avg_undiscounted_return(expert_trajs)
print('The expert ' +
      'reached the goal in ' + str(expert_performance) + ' of trajs.')

meirl = MaxEnt(env, expert_trajs=expert_trajs, transition_dynamics=transition_dynamics, rl_alg_factory=rl_alg_factory, learning_rate=1e-1)
print('Running MaxEnt IRL...')
rewards = meirl.train(feat_map, time_limit=120, verbose=False)
rewards = (rewards - np.min(rewards))/(np.max(rewards) - np.min(rewards))

print('Normalized reward coefficients returned by MaxEnt IRL:')
print(rewards)

# Check how often the goal state was reached.
test_agent = rl_alg_factory(meirl.env)
print('Training test agent w/ reward obtained by MaxEnt...')
test_agent.train(30)
print('Done training test agent.')
test_trajs = collect_trajs(meirl.env, test_agent, no_episodes,
                           max_steps_per_episode,
                           store_to='data/frozen/foo')
irl_performance = avg_undiscounted_return(test_trajs)
print('The test agent trained w/ the reward estimated by MaxEnt IRL ' +
      'reached the goal in ' + str(irl_performance) + ' of trajs.')
