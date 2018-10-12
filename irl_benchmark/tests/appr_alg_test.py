'''Test both implementations of Apprenticeship IRL (Abeel & Ng, 2004).'''

import gym
import numpy as np

from irl_benchmark.irl.algorithms.appr.appr_irl import ApprIRL
from irl_benchmark.irl.feature.feature_wrapper import FrozenLakeFeatureWrapper
from irl_benchmark.irl.collect import collect_trajs
from irl_benchmark.irl.reward.reward_function import FeatureBasedRewardFunction
from irl_benchmark.irl.reward.reward_wrapper import RewardWrapper
from irl_benchmark.rl.algorithms.value_iteration import ValueIteration
from irl_benchmark.utils.utils import avg_undiscounted_return


def rl_alg_factory(env):
    return ValueIteration(env, error=1e-5)


def run_appr_irl(irl_train_time=20, large_map=False,
                 use_projection=False, no_episodes=1000, horizon=100,
                 store_to='data/frozen/appr'):
    '''Run ApprIRL on FrozenLake and return distance of feature counts.

    The RL steps use Value Iteration.

    Args:
      use_projection: `bool`, use projection alg if true, else use max-margin
      duration: `float`, maximum total training time for IRL and initial expert

    Returns:
      appr_irl.distances: `list`, L2 distances betw expert feature counts and
                          the feature counts obtained by the current RL agent
                          (trained using the current reward estimate).
    '''
    # For FrozenLake8x8 we need longer trajectories.
    if large_map:
        horizon *= 5

    # Get expert trajectories.
    # Value iteration solves FrozenLake and FrozenLake8x8 in 15 seconds.
    expert_store = store_to + '/expert'
    if large_map:
        env = gym.make('FrozenLake8x8-v0')
    else:
        env = gym.make('FrozenLake-v0')
    n_states = env.observation_space.n
    env = FrozenLakeFeatureWrapper(env)
    expert_agent = ValueIteration(env, error=1e-5)
    expert_agent.train(15)
    expert_trajs = collect_trajs(env, expert_agent, no_episodes,
                                 horizon, expert_store)

    # Run Apprenticeship IRL and agent trained w/ the obtained reward.
    initial_reward = np.random.normal(size=16)
    reward_function = FeatureBasedRewardFunction(env,
                                                 parameters=initial_reward)
    env = RewardWrapper(env, reward_function)
    appr_irl = ApprIRL(env, expert_trajs, rl_alg_factory, proj=use_projection)
    appr_irl.train(time_limit=irl_train_time,
                   rl_time_per_iteration=15, verbose=False)
    test_agent = ValueIteration(appr_irl.env, error=1e-5)
    test_agent.train(15)
    test_store = store_to + '/test'
    test_trajs = collect_trajs(appr_irl.env, test_agent, no_episodes,
                               horizon, test_store)

    # Check how often the goal state was reached.
    expert_performance = avg_undiscounted_return(expert_trajs)
    irl_performance = avg_undiscounted_return(test_trajs)

    results = {
        'irl_reward': appr_irl.env.reward_function.parameters,
        'initial_reward': initial_reward,
        'irl_values': test_agent.V,
        'irl_policy': [test_agent.policy(s) for s in range(n_states)],
        'distances': appr_irl.distances,
        'expert_avg_return': expert_performance,
        'test_avg_return': irl_performance,
        }
    return results


def test_svm(duration=2):
    '''Test if the max-margin implementation plausibly works.

    Checks if it runs for a credible number of steps, and if the latest
    distance is credibly small, and strictly smaller than the first.

    NOTE: With the default value of duration, this test only checks if
    ApprIRL compiles at all, not if its results are credible.
    '''
    # In all cases check whether ApprIRL runs at all.
    results = run_appr_irl(irl_train_time=duration, use_projection=False)
    distances = results['distances']

    # Check results only if duration was manually set to be large.
    if duration < 5:
        return
    assert len(distances) >= 3
    assert len(distances) <= 10  # unrealistically high
    assert distances[-1] < distances[0]
    assert distances[-1] < 5


def test_proj(duration=2):
    '''Test if the projection implementation plausibly works.

    Checks if it runs for a credible number of steps, and if the latest
    distance is credibly small, and strictly smaller than the first.

    NOTE: With the default value of duration, this test only checks if
    ApprIRL compiles at all, not if its results are credible.
    '''
    # In all cases check whether ApprIRL runs at all.
    results = run_appr_irl(irl_train_time=duration, use_projection=True)
    distances = results['distances']

    # Check results only if duration was manually set to be large.
    if duration < 5:
        return
    assert len(distances) >= 3
    assert len(distances) <= 10  # unrealistically high
    assert distances[-1] < distances[0]
    assert distances[-1] < 5


svm_test_avg_results = []
# svm_exp_avg_rets = []
proj_test_avg_results = []
# proj_exp_avg_rets = []
for i in range(10):
    svm_test_avg_result = run_appr_irl(no_episodes=100000)
    # svm_exp_avg_ret = run_appr_irl()['expert_avg_return']
    proj_test_avg_result = run_appr_irl(use_projection=True,
                                        no_episodes=100000)
    # proj_exp_avg_ret = run_appr_irl(use_projection=True)['expert_avg_return']
    svm_test_avg_results.append(svm_test_avg_result)
    # svm_exp_avg_rets.append(svm_exp_avg_ret)
    proj_test_avg_results.append(proj_test_avg_result)
    # proj_exp_avg_rets.append(proj_exp_avg_ret)
# print('SVM test avg returns')
# print(svm_test_avg_returns)
# # print('SVM expert avg returns')
# # print(svm_exp_avg_rets)
# print('\n')
# print('Proj test avg returns')
# print(proj_test_avg_returns)
# print('Proj expert avg returns')
# print(proj_exp_avg_rets)
# print('\n')
