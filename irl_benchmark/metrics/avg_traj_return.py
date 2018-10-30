from irl_benchmark.irl.collect import collect_trajs
from irl_benchmark.metrics.base_metric import BaseMetric
from irl_benchmark.utils.rl import true_reward_per_traj


class AverageTrajectoryReturn(BaseMetric):
    def __init__(self, metric_input: dict):
        assert 'env' in metric_input.keys()
        self.env = metric_input['env']

    def evaluate(self, evaluation_input: dict):
        assert 'irl_agent' in evaluation_input.keys()
        irl_trajs = self.generate_traj_if_not_exists(evaluation_input)
        return true_reward_per_traj(irl_trajs)
