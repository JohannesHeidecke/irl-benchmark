"""Module for average trajectory return metric."""

from irl_benchmark.metrics.base_metric import BaseMetric
from irl_benchmark.utils.rl import true_reward_per_traj


class AverageTrajectoryReturn(BaseMetric):
    """Average sum of true rewards per trajectory."""

    def __init__(self, metric_input: dict):
        assert 'env' in metric_input.keys()
        self.env = metric_input['env']

    def evaluate(self, evaluation_input: dict):
        """Evaluate the metric given some input and return result.

        Parameters
        ----------
        evaluation_input: dict
            The evaluation input. Requires 'irl_agent' field.

        Returns
        -------
        float
            The average sum of true rewards per trajectory.
        """
        assert 'irl_agent' in evaluation_input.keys()
        irl_trajs = self.generate_traj_if_not_exists(evaluation_input)
        return true_reward_per_traj(irl_trajs)
