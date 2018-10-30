import numpy as np

from irl_benchmark.irl.collect import collect_trajs
from irl_benchmark.metrics.base_metric import BaseMetric
from irl_benchmark.utils.irl import feature_count


class FeatureCountInfLoss(BaseMetric):
    def __init__(self, metric_input: dict):
        assert 'env' in metric_input.keys()
        assert 'expert_trajs' in metric_input.keys()
        super(FeatureCountInfLoss, self).__init__(metric_input)
        expert_trajs = metric_input['expert_trajs']
        self.env = metric_input['env']
        self.expert_feature_count = feature_count(
            self.env, expert_trajs, gamma=1.0)

    def evaluate(self, evaluation_input: dict):
        assert 'irl_agent' in evaluation_input.keys()
        irl_trajs = self.generate_traj_if_not_exists(evaluation_input)
        irl_feature_count = feature_count(self.env, irl_trajs, gamma=1.0)
        diff = self.expert_feature_count - irl_feature_count
        return np.linalg.norm(diff, ord=np.inf)
