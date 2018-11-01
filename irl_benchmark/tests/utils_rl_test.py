from irl_benchmark.utils.rl import true_reward_per_traj


def test_true_reward_per_traj():
    # create dummy data:
    trajs = [{'true_rewards': [1.0, 2.0, 0.0]}, {'true_rewards': [2.0]}]
    assert true_reward_per_traj(trajs) == 2.5
    assert true_reward_per_traj(trajs[:1])
    assert true_reward_per_traj(trajs[-1:]) == 2.0
    # empty trajectories should have zero reward
    trajs = [{'true_rewards': []}]
    assert true_reward_per_traj(trajs) == 0
