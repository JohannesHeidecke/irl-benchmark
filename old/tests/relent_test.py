'''Test Relative Entropy IRL (Boularias et al., 2011).'''

import gym
import numpy as np

from irl_benchmark.irl.algorithms.relent.relent import RelEnt
from irl_benchmark.irl.feature.feature_wrapper import FrozenLakeFeatureWrapper
from irl_benchmark.irl.feature.feature_wrapper import FeatureWrapper
from irl_benchmark.irl.collect import collect_trajs
from irl_benchmark.irl.reward.reward_function import FeatureBasedRewardFunction
from irl_benchmark.irl.reward.reward_wrapper import RewardWrapper
from irl_benchmark.rl.algorithms.value_iteration import ValueIteration
from irl_benchmark.utils.utils import unwrap_env, avg_undiscounted_return


def rl_alg_factory(env, lp=False):
    '''Return an RL algorithm that will collect expert trajectories.

    In this module, this always returns a Value Iteration agent.
    '''
    return ValueIteration(env, error=1e-5)


def run_relent_irl(irl_step_size=1e-3, irl_train_time=60,
                   rl_train_time=15, large_map=False,
                   no_episodes=1000, horizon=100,
                   store_to='data/frozen/relent'):
    '''Run RelEnt IRL on FrozenLake and return dictionary w/ metrics.

    Value iteration is used for the expert and the test agent.

    Args:
      irl_step_size: `float`, step size for RelEnt IRL's gradient ascent
      irl_train_time: `float`, training time for RelEnt IRL in seconds
      large_map: `bool`, use FrozenLake8x8 instead of 4x4 if true
      no_episodes: `int`, number of episodes collected by expert, RelEnt IRL's
                   baseline policy, and the test agent trained w/ the reward
                   obtained by RelEnt IRL
      horizon: `int`, maximum length of each episode

    Returns `dictionary` results w/ keys:
    'irl_reward' -- `np.ndarray` of shape (16,) or (64,) (for FrozenLake8x8),
                    reward coefficients of the reward obtained by RelEnt IRL
    'initial_reward' -- `np.ndarray` of shape (16,) or (64,), random reward
                        coefficients used to initialize gradient ascent
    'expert_avg_return' -- `float`, expert probability of reaching goal
    'test_avg_return' -- `float`, probability that a test agent trained w/
                         the reward obtained by RelEnt reaches the goal
    '''
    # Get expert trajectories.
    # Value iteration solves FrozenLake and FrozenLake8x8 in 15 seconds.
    if large_map:
        env = gym.make('FrozenLake8x8-v0')
    else:
        env = gym.make('FrozenLake-v0')
    env = FrozenLakeFeatureWrapper(env)
    expert_agent = rl_alg_factory(env)
    expert_agent.train(rl_train_time)
    expert_store = store_to + '/expert'
    expert_trajs = collect_trajs(env, expert_agent, no_episodes,
                                 horizon, expert_store)

    # Provide random reward function as initial reward estimate.
    # This probably isn't really required.
    n_features = unwrap_env(env, FeatureWrapper).feature_shape()[0]
    initial_reward_coefficients = np.random.normal(size=n_features)
    reward_function = FeatureBasedRewardFunction(env,
                                                 initial_reward_coefficients)
    env = RewardWrapper(env, reward_function)

    # Train Relative Entropy IRL and test_agent based on the obtained reward.
    relent = RelEnt(env, expert_trajs, rl_alg_factory,
                    horizon=horizon, eps=1e-3)
    relent.train(irl_step_size, irl_train_time, verbose=False)
    test_agent = rl_alg_factory(relent.env)
    test_agent.train(60)
    test_store = store_to + '/test'
    test_trajs = collect_trajs(relent.env, test_agent, no_episodes,
                               horizon, test_store)

    # Check how often the goal state was reached.
    expert_performance = avg_undiscounted_return(expert_trajs)
    irl_performance = avg_undiscounted_return(test_trajs)

    results = {
        'irl_reward': relent.env.reward_function.parameters,
        'initial_reward': initial_reward_coefficients,
        'expert_avg_return': expert_performance,
        'test_avg_return': irl_performance,
        }
    return results


def test_relent(duration=1, no_episodes=1, horizon=1):
    '''Test if RelEnt IRL plausibly works.

    Checks if:
    - An agent trained w/ the obtained reward reaches the goal
    in more than 1% of trajectories; and
    - if the obtained reward differs from the randomly initialized one.

    NOTE: With the default value of duration, this test only checks if
    ApprIRL compiles at all, not if its results are credible.
    '''
    # In all cases check whether ApprIRL runs at all.
    results = run_relent_irl(irl_train_time=duration,
                             rl_train_time=duration,
                             no_episodes=no_episodes, horizon=horizon)

    # Check results only if duration was manually set to be large.
    if duration > 5:
        assert np.linalg.norm(
            results['initial_reward'] - results['irl_reward']
            ) > .1  # Note that irl_reward is normalized to a unit vector.
        assert results['test_avg_return'] > .01
