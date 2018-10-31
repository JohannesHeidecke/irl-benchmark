"""Util functions related to IRL."""

from typing import Dict, List
import numpy as np

from irl_benchmark.irl.feature.feature_wrapper import FeatureWrapper
from irl_benchmark.utils.wrapper import unwrap_env, is_unwrappable_to


def feature_count(env, trajs: List[Dict[str, list]],
                  gamma: float) -> np.ndarray:
    """Return empirical discounted feature counts of input trajectories.

    Parameters
    ----------
    env: gym.Env
        A gym environment, wrapped in a feature wrapper
    trajs: List[Dict[str, list]]
         A list of trajectories.
        Each trajectory is a dictionary with keys
        ['states', 'actions', 'rewards', 'true_rewards', 'features'].
        The values of each dictionary are lists.
        See :func:`irl_benchmark.irl.collect.collect_trajs`.
    gamma: float
        The discount factor. Must be in range [0., 1.].

    Returns
    -------
    np.ndarray
        A numpy array containing discounted feature counts. The shape
        is the same as the trajectories' feature shapes. One scalar
        feature count per feature.
    """
    assert is_unwrappable_to(env, FeatureWrapper)

    # Initialize feature count sum to zeros of correct shape:
    feature_count_sum = np.zeros(
        unwrap_env(env, FeatureWrapper).feature_shape())

    for traj in trajs:
        assert traj['features']  # empty lists are False in python

        # gammas is a vector containing [gamma^0, gamma^1, gamma^2, ... gamma^l]
        # where l is length of the trajectory:
        gammas = gamma**np.arange(len(traj['features']))
        traj_feature_count = np.sum(
            gammas.reshape(-1, 1) * np.array(traj['features']), axis=0)
        # add trajectory's feature count:
        feature_count_sum += traj_feature_count
    # divide feature_count_sum by number of trajectories to normalize:
    result = feature_count_sum / len(trajs)
    return result
