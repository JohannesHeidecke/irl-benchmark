import os
import gym
import numpy as np

from irl_benchmark.rl.algorithms.tabular_q import TabularQ
from irl_benchmark.utils.utils import MetricsLogger

from comet_ml import Experiment


def run():
    if 'COMET_KEY' not in os.environ:
        print('Please set the COMET_KEY environment variable')
        print('e.g. by running with: `COMET_KEY="your-key" python example.py`')
        return
    api_key = os.environ['COMET_KEY']
    # Create an experiment
    experiment = Experiment(
        api_key=api_key, project_name="tabular_q", workspace="irl-benchmark")

    metrics_logger = MetricsLogger()

    def metric_listener(reward):
        experiment.log_metric('reward', reward)
        metrics_logger.log_metric('reward', reward)

    duration = 5
    hyper_params = {'gamma': 0.95, 'alpha_start': 0.8}

    # Report hparams
    experiment.log_multiple_params(hyper_params)

    env = gym.make('FrozenLake-v0')
    agent = TabularQ(env, **hyper_params)
    rewards = agent.train(duration, metric_listener)

    experiment.log_metric('MA(100) reward', np.mean(rewards[-100:]))

    metrics_logger.save('data/example_tabular_q_metrics.json')


if __name__ == '__main__':
    run()
