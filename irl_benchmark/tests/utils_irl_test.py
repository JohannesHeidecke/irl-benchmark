import numpy as np

from irl_benchmark.irl.feature import feature_wrapper
from irl_benchmark.utils.irl import feature_count
from irl_benchmark.utils.general import to_one_hot


def test_feature_count():
    env = feature_wrapper.make('FrozenLake-v0')
    # create dummy data:
    path = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]
    features = []
    for i in path:
        features.append(to_one_hot(i, 16))

    trajs = [{'features': features}]
    result = feature_count(env, trajs, gamma=1.0)
    desired = np.array(
        [0., 1., 2., 3., 4., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    assert isinstance(result, np.ndarray)
    assert np.allclose(result, desired)

    # two times the same traj should get the same feature count:
    trajs = [{'features': features}, {'features': features}]
    result = feature_count(env, trajs, gamma=1.0)
    desired = np.array(
        [0., 1., 2., 3., 4., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    assert isinstance(result, np.ndarray)
    assert np.allclose(result, desired)

    # repeating a traj twice should double feature count (with gamma 1)
    trajs = [{'features': features + features}]
    result = feature_count(env, trajs, gamma=1.0)
    desired = np.array(
        [0., 2., 4., 6., 8., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    assert isinstance(result, np.ndarray)
    assert np.allclose(result, desired)

    # test gamma 0.9:
    trajs = [{'features': features}]
    result = feature_count(env, trajs, gamma=.9)
    x = .9**6 + .9**7 + .9**8 + .9**9
    desired = np.array([
        0., 1., .9 + .81, .729 + .6561 + .59049, x, 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0.
    ])
    assert isinstance(result, np.ndarray)
    assert np.allclose(result, desired)

    # test gamma 0:
    result = feature_count(env, trajs, gamma=0)
    desired = np.array(
        [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    assert isinstance(result, np.ndarray)
    assert np.allclose(result, desired)
