from gym.envs.toy_text.frozen_lake import FrozenLakeEnv
import numpy as np

from irl_benchmark.envs import make_env, make_wrapped_env
from irl_benchmark.envs.maze_world import MazeWorld, get_maps, MAP0, MAP1
from irl_benchmark.irl.feature.feature_wrapper import FeatureWrapper
from irl_benchmark.irl.reward.reward_function import FeatureBasedRewardFunction
from irl_benchmark.irl.reward.reward_wrapper import RewardWrapper
from irl_benchmark.rl.model.model_wrapper import BaseWorldModelWrapper
from irl_benchmark.utils.wrapper import is_unwrappable_to, unwrap_env


def test_make_frozen():
    env = make_env('FrozenLake-v0')
    assert is_unwrappable_to(env, FrozenLakeEnv)


def test_make_frozen8():
    env = make_env('FrozenLake8x8-v0')
    assert is_unwrappable_to(env, FrozenLakeEnv)


def test_make_maze0():
    env = make_env('MazeWorld0-v0')
    assert is_unwrappable_to(env, MazeWorld)
    walls, rews = get_maps(MAP0)
    maze_env = unwrap_env(env, MazeWorld)
    assert np.all(maze_env.map_walls == walls)
    assert np.all(maze_env.map_rewards == rews)


def test_make_maze1():
    env = make_env('MazeWorld1-v0')
    assert is_unwrappable_to(env, MazeWorld)
    walls, rews = get_maps(MAP1)
    maze_env = unwrap_env(env, MazeWorld)
    assert np.all(maze_env.map_walls == walls)
    assert np.all(maze_env.map_rewards == rews)


def case_make_wrapped(env_id):
    env = make_wrapped_env(env_id)
    assert not is_unwrappable_to(env, FeatureWrapper)
    assert not is_unwrappable_to(env, RewardWrapper)
    assert not is_unwrappable_to(env, BaseWorldModelWrapper)

    env = make_wrapped_env(env_id, with_feature_wrapper=True)
    assert is_unwrappable_to(env, FeatureWrapper)
    assert not is_unwrappable_to(env, RewardWrapper)
    assert not is_unwrappable_to(env, BaseWorldModelWrapper)

    env = make_wrapped_env(
        env_id, with_feature_wrapper=True, with_model_wrapper=True)
    assert is_unwrappable_to(env, FeatureWrapper)
    assert not is_unwrappable_to(env, RewardWrapper)
    assert is_unwrappable_to(env, BaseWorldModelWrapper)

    def rew_fun_fact(env):
        return FeatureBasedRewardFunction(env, 'random')

    env = make_wrapped_env(
        env_id,
        with_feature_wrapper=True,
        reward_function_factory=rew_fun_fact,
        with_model_wrapper=False)
    assert is_unwrappable_to(env, FeatureWrapper)
    assert is_unwrappable_to(env, RewardWrapper)
    assert not is_unwrappable_to(env, BaseWorldModelWrapper)

    env = make_wrapped_env(
        env_id,
        with_feature_wrapper=True,
        reward_function_factory=rew_fun_fact,
        with_model_wrapper=True)
    assert is_unwrappable_to(env, FeatureWrapper)
    assert is_unwrappable_to(env, RewardWrapper)
    assert is_unwrappable_to(env, BaseWorldModelWrapper)


def test_make_wrapped_frozen():
    case_make_wrapped('FrozenLake-v0')


def test_make_wrapped_frozen8():
    case_make_wrapped('FrozenLake8x8-v0')


def test_make_wrapped_maze0():
    case_make_wrapped('MazeWorld0-v0')


def test_make_wrapped_maze1():
    case_make_wrapped('MazeWorld1-v0')
