# import gym
import numpy as np

# from irl_benchmark.irl.feature.feature_wrapper import FrozenLakeFeatureWrapper, unwrap_env
from irl_benchmark.irl.feature import feature_wrapper
from irl_benchmark.irl.reward.reward_function import FeatureBasedRewardFunction
from irl_benchmark.irl.reward.reward_wrapper import RewardWrapper
from irl_benchmark.rl.model.discrete_env import DiscreteEnvModelWrapper

env = feature_wrapper.make('FrozenLake-v0')
env = DiscreteEnvModelWrapper(env)
a = FeatureBasedRewardFunction(env, parameters=np.random.rand(16))
env = RewardWrapper(env, a)
from irl_benchmark.irl.collect import collect_trajs
from irl_benchmark.rl.algorithms.value_iteration import ValueIteration


def rl_alg_factory(env):
    return ValueIteration(env, {'gamma': 0.9})


expert_agent = rl_alg_factory(env)
expert_agent.train(None)
expert_trajs = collect_trajs(
    env, expert_agent, 10000, None, verbose=True)
from irl_benchmark.irl.algorithms.lp_irl import LinProgIRL
from irl_benchmark.irl.collect import load_stored_trajs

expert_trajs = load_stored_trajs('data/frozen/expert/')
from irl_benchmark.metrics.avg_traj_return import AverageTrajectoryReturn
from irl_benchmark.metrics.feature_count_l2 import FeatureCount2Loss
from irl_benchmark.metrics.feature_count_inf import FeatureCountInfLoss

lp = LinProgIRL(env, expert_trajs, rl_alg_factory=ValueIteration,
                metrics=[AverageTrajectoryReturn, FeatureCount2Loss, FeatureCountInfLoss], config=None)

alphas = lp.train(1000, 100, 1000)

print(alphas)
