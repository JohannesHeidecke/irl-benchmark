import numpy as np

from irl_benchmark.envs import make_wrapped_env
from irl_benchmark.envs.maze_world import MazeWorld
from irl_benchmark.irl.feature.feature_wrapper import FeatureWrapper
from irl_benchmark.utils.wrapper import unwrap_env


def test_frozen_features():
    env = make_wrapped_env('FrozenLake-v0', with_feature_wrapper=True)
    feature_wrapper = unwrap_env(env, FeatureWrapper)
    d = feature_wrapper.feature_dimensionality()
    ranges = feature_wrapper.feature_range()
    print(ranges)
    for i in range(16):
        feature = feature_wrapper.features(None, None, i)
        assert feature.shape == d
        assert np.all(feature >= ranges[0])
        assert np.all(feature <= ranges[1])


def test_frozen8_features():
    env = make_wrapped_env('FrozenLake8x8-v0', with_feature_wrapper=True)
    feature_wrapper = unwrap_env(env, FeatureWrapper)
    d = feature_wrapper.feature_dimensionality()
    ranges = feature_wrapper.feature_range()
    print(ranges)
    for i in range(64):
        feature = feature_wrapper.features(None, None, i)
        assert feature.shape == d
        assert np.all(feature >= ranges[0])
        assert np.all(feature <= ranges[1])


def test_maze0_features():
    env = make_wrapped_env('MazeWorld0-v0', with_feature_wrapper=True)
    maze_env = unwrap_env(env, MazeWorld)
    feature_wrapper = unwrap_env(env, FeatureWrapper)
    d = feature_wrapper.feature_dimensionality()
    ranges = feature_wrapper.feature_range()
    print(ranges)
    for i in range(10240, 13):
        for a in range(10):
            feature = feature_wrapper.features(
                maze_env.index_to_state(i), a, None)
            assert feature.shape == d
            assert np.all(feature >= ranges[0])
            assert np.all(feature <= ranges[1])


def test_maze1_features():
    env = make_wrapped_env('MazeWorld1-v0', with_feature_wrapper=True)
    maze_env = unwrap_env(env, MazeWorld)
    feature_wrapper = unwrap_env(env, FeatureWrapper)
    d = feature_wrapper.feature_dimensionality()
    ranges = feature_wrapper.feature_range()
    print(ranges)
    for i in range(10240, 13):
        for a in range(10):
            feature = feature_wrapper.features(
                maze_env.index_to_state(i), a, None)
            assert feature.shape == d
            assert np.all(feature >= ranges[0])
            assert np.all(feature <= ranges[1])
