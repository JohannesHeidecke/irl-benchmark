from abc import ABC, abstractmethod


class BaseMetric(ABC):
    def __init__(self, metric_input: dict = None):
        self.metric_input = metric_input

    @abstractmethod
    def evaluate(self, evaluation_input: dict = None) -> float:
        raise NotImplementedError()
