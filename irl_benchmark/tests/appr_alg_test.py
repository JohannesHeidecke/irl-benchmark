import gym
import numpy as np

import pytest

from irl_benchmark.irl.algorithms.appr.appr_irl import ApprIRL
from irl_benchmark.irl.feature.feature_wrapper import FrozenLakeFeatureWrapper
from irl_benchmark.irl.collect import collect_trajs
from irl_benchmark.irl.reward.reward_function import FeatureBasedRewardFunction
from irl_benchmark.irl.reward.reward_wrapper import RewardWrapper
from irl_benchmark.rl.algorithms import TabularQ
'''Test both implementations of Apprenticeship IRL (Abeel & Ng, 2004).'''


def run_appr_irl(use_projection, duration):
    '''Run ApprIRL on FrozenLake and return distance of feature counts.

    The RL steps use Tabular Q-learning.

    Args:
      use_projection: `bool`, use projection alg if true, else use max-margin
      duration: `float`, maximum total training time for IRL and initial expert

    Returns:
      appr_irl.distances: `list`, L2 distances betw expert feature counts and
                          the feature counts obtained by the current RL agent
                          (trained using the current reward estimate).
    '''
    store_to = 'data/frozen/expert'
    no_episodes = 1000
    max_steps_per_episode = 100
    env = gym.make('FrozenLake-v0')
    env = FrozenLakeFeatureWrapper(env)
    expert_agent = TabularQ(env)
    expert_agent.train(duration)
    expert_trajs = collect_trajs(env, expert_agent, no_episodes,
                                 max_steps_per_episode, store_to)
    reward_function = FeatureBasedRewardFunction(env,
                                                 np.random.normal(size=16))
    env = RewardWrapper(env, reward_function)
    appr_irl = ApprIRL(env, expert_trajs, TabularQ, proj=use_projection)
    appr_irl.train(
        time_limit=duration, rl_time_per_iteration=duration / 2, verbose=False)
    return appr_irl.distances


def test_svm(duration=0.1):
    '''Test if the max-margin implementation plausibly works.

    Checks if it runs for a credible number of steps, and if the latest
    distance is credibly small, and strictly smaller than the first.

    NOTE: With the default value of duration, this test only checks if
    ApprIRL compiles at all, not if its results are credible.
    '''
    # In all cases check whether ApprIRL runs at all.
    distances = run_appr_irl(False, duration)

    # Check results only if duration was manually set to be large.
    if duration < 5:
        return
    assert len(distances) >= 2
    assert len(distances) <= 10  # unrealistically high
    assert distances[-1] < distances[0]
    assert distances[-1] < 5


def test_proj(duration=0.1):
    '''Test if the projection implementation plausibly works.

    Checks if it runs for a credible number of steps, and if the latest
    distance is credibly small, and strictly smaller than the first.

    NOTE: With the default value of duration, this test only checks if
    ApprIRL compiles at all, not if its results are credible.
    '''
    # In all cases check whether ApprIRL runs at all.
    distances = run_appr_irl(True, duration)

    # Check results only if duration was manually set to be large.
    if duration < 5:
        return
    assert len(distances) >= 2
    assert len(distances) <= 10  # unrealistically high
    assert distances[-1] < distances[0]
    assert distances[-1] < 5


@pytest.mark.slow
def test_svm_slow():
    test_svm(20)


@pytest.mark.slow
def test_proj_slow():
    test_proj(20)
