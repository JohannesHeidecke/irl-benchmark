import os
import sys

import gym
import numpy as np

from irl_benchmark.rl.algorithms.tabular_q import TabularQ

from comet_ml import Experiment
'''
This runs a load testing of comet.ml as well.
It also tests if it is enough to optimise each hyper_param independently.
'''


def run(**hyper_params):
    if 'COMET_KEY' not in os.environ:
        print('Please set the COMET_KEY environment variable')
        print('e.g. by running with: `COMET_KEY="your-key" python example.py`')
        sys.exit()
    api_key = os.environ['COMET_KEY']
    # Create an experiment
    experiment = Experiment(
        api_key=api_key, project_name="tabular_q", workspace="irl-benchmark")

    def metric_listener(reward):
        experiment.log_metric('reward', reward)

    duration = 5

    # Report hparams
    experiment.log_multiple_params(hyper_params)

    env = gym.make('FrozenLake-v0')
    agent = TabularQ(env, **hyper_params)

    rewards = agent.train(duration, metric_listener)
    ma_100 = np.mean(rewards[-100:])
    experiment.log_metric('MA(100) reward', ma_100)

    print('Params:', hyper_params, 'got ma_100 reward', ma_100)
    return rewards


if __name__ == '__main__':
    # Parameter search
    N = 5
    grid = {
        'gamma': np.geomspace(0.5, 1.0, N),
        'alpha_start': np.geomspace(2.0, 0.5, N),
        'alpha_end': np.geomspace(1.0, 0.05, N),
        'alpha_decay': np.geomspace(1e-5, 1e-3, N),
        'eps_start': np.geomspace(1.0, 0.5, N),
        'eps_end': np.geomspace(0.1, 0.001, N),
        'eps_decay': np.geomspace(1e-5, 1e-3, N),
    }

    # Default values:
    run()

    # Pick middle value as init:
    params = {k: v[N // 2] for k, v in grid.items()}
    run(**params)

    for _ in range(2):
        # Optimise each parameter independently M times
        for k, ps in grid.items():
            best_p = None
            best_r = 0.0

            # Loop over params
            for p in ps:
                params[k] = p
                r = np.mean(run(**params)[-100:])
                if best_p is None or r > best_r:
                    best_p = p
                    best_r = r
            params[k] = best_p
            print('Best {} was {} with r = {}'.format(k, best_p, best_r))
        print('Final params:', params)
