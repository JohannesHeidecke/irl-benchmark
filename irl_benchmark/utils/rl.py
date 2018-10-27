"""Utils related to reinforcement learning."""
from typing import Dict, List

import numpy as np


def true_reward_per_traj(trajs: List[Dict[str, list]]) -> float:
    """Return (undiscounted) average sum of true rewards per trajectory.

    Parameters
    ----------
    trajs: List[Dict[str, list]])
        A list of trajectories.
        Each trajectory is a dictionary with keys
        ['states', 'actions', 'rewards', 'true_rewards', 'features'].
        The values of each dictionary are lists.
        See :func:`irl_benchmark.irl.collect.collect_trajs`.

    Returns
    -------
    float
        The undiscounted average sum of true rewards per trajectory.

    """
    true_reward_sum = 0
    for traj in trajs:
        true_reward_sum += np.sum(traj['true_rewards'])
    return true_reward_sum / len(trajs)
