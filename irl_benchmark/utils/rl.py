"""Utils related to reinforcement learning."""

import numpy as np


def true_reward_per_traj(trajs):
    """Return (undiscounted) average sum of true rewards per trajectory."""
    # TODO: add docstrings and typing
    true_reward_sum = 0
    for traj in trajs:
        true_reward_sum += np.sum(traj['true_rewards'])
    return true_reward_sum / len(trajs)
