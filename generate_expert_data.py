import numpy as np

from irl_benchmark.envs import make_wrapped_env
from irl_benchmark.irl.collect import collect_trajs
from irl_benchmark.irl.feature import feature_wrapper
from irl_benchmark.rl.algorithms.value_iteration import ValueIteration
from irl_benchmark.rl.model.maze_world import MazeModelWrapper
from irl_benchmark.utils.wrapper import unwrap_env

# Run this script to generate all expert data.

# # FROZEN LAKE:
# env = feature_wrapper.make('FrozenLake-v0')
# def rl_alg_factory(env):
#     return ValueIteration(env, {'gamma': 0.9})
# expert_agent = rl_alg_factory(env)
# expert_agent.train(None)
# expert_trajs = collect_trajs(env, expert_agent, 10000, None, 'data/frozen/expert/', verbose=True)
#
# # FROZEN LAKE 8x8:
# env = feature_wrapper.make('FrozenLake8x8-v0')
# def rl_alg_factory(env):
#     return ValueIteration(env, {'gamma': 0.9})
# expert_agent = rl_alg_factory(env)
# expert_agent.train(None)
# expert_trajs = collect_trajs(env, expert_agent, 10000, None, 'data/frozen8/expert/', verbose=True)


# MAZE WORLD 0:
def rl_alg_factory(env):
    class FakeHuman():
        def __init__(self, env):
            self.model_wrapper = unwrap_env(env, MazeModelWrapper)
            self.policy = np.load('data/maze0/expert_fakehuman/human_policy1.npy')


        def pick_action(self, state):
            state_index = self.model_wrapper.state_to_index(state)
            policy = self.policy[state_index, :]
            return int(np.random.choice(10, p=policy))

        def train(self, duration=None):
            pass

    return FakeHuman(env)
    # return ValueIteration(env, {'gamma': 1.0})

env = make_wrapped_env('MazeWorld0-v0', True, None, True)
expert_agent = rl_alg_factory(env)
expert_agent.train(None)
expert_trajs = collect_trajs(env, expert_agent, 100000, 200, 'data/maze0/expert_fakehuman/', verbose=True)

