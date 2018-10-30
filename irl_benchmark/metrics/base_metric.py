from abc import ABC, abstractmethod

from irl_benchmark.irl.collect import collect_trajs


class BaseMetric(ABC):
    def __init__(self, metric_input: dict = None):
        self.metric_input = metric_input

    @abstractmethod
    def evaluate(self, evaluation_input: dict = None) -> float:
        raise NotImplementedError()

    def generate_traj_if_not_exists(self, evaluation_input: dict):
        assert 'irl_agent' in evaluation_input.keys()
        if not 'irl_trajs' in evaluation_input:
            print('generating new trajs for metrics')
            evaluation_input['irl_trajs'] = collect_trajs(self.env, evaluation_input['irl_agent'], 100)
        else:
            print('reuse generated trajs for metric')
        return evaluation_input['irl_trajs']
