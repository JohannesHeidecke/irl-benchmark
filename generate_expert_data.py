
from irl_benchmark.irl.collect import collect_trajs
from irl_benchmark.irl.feature import feature_wrapper
from irl_benchmark.rl.algorithms.value_iteration import ValueIteration

# Run this script to generate all expert data.

# FROZEN LAKE:
env = feature_wrapper.make('FrozenLake-v0')
def rl_alg_factory(env):
    return ValueIteration(env, {'gamma': 0.9})
expert_agent = rl_alg_factory(env)
expert_agent.train(None)
expert_trajs = collect_trajs(env, expert_agent, 10000, None, 'data/frozen/expert/', verbose=True)

# FROZEN LAKE 8x8:
env = feature_wrapper.make('FrozenLake8x8-v0')
def rl_alg_factory(env):
    return ValueIteration(env, {'gamma': 0.9})
expert_agent = rl_alg_factory(env)
expert_agent.train(None)
expert_trajs = collect_trajs(env, expert_agent, 10000, None, 'data/frozen8/expert/', verbose=True)