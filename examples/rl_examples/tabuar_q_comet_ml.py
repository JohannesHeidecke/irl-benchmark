import gym
import numpy as np

from irl_benchmark.rl.algorithms.tabular_q import TabularQ

from comet_ml import Experiment


def run():
    # Create an experiment
    experiment = Experiment(
        api_key="lXbOUQyoigKWUW6OrLSoTiJyY",
        project_name="tabular_q",
        workspace="irl-benchmark")

    def metric_listener(reward):
        experiment.log_metric('reward', reward)

    duration = 5
    hyper_params = {'gamma': 0.95, 'alpha_start': 0.8}

    # Report hparams
    experiment.log_multiple_params(hyper_params)

    env = gym.make('FrozenLake-v0')
    agent = TabularQ(env, **hyper_params)
    rewards = agent.train(duration, metric_listener)

    experiment.log_metric('MA(100) reward', np.mean(rewards[-100:]))


if __name__ == '__main__':
    run()
