import numpy as np

from irl_benchmark.irl.algorithms.appr_irl import ApprIRL
from irl_benchmark.irl.algorithms.mce_irl import MaxCausalEntIRL
from irl_benchmark.irl.algorithms.me_irl import MaxEntIRL
from irl_benchmark.irl.feature import feature_wrapper
from irl_benchmark.irl.reward.reward_function import FeatureBasedRewardFunction
from irl_benchmark.irl.reward.reward_wrapper import RewardWrapper
from irl_benchmark.rl.algorithms import ValueIteration


def quick_run_alg(alg_class, config={}):
    env = feature_wrapper.make('FrozenLake-v0')
    reward_function = FeatureBasedRewardFunction(env, 'random')
    env = RewardWrapper(env, reward_function)

    def rl_alg_factory(env):
        return ValueIteration(env, {})

    expert_trajs = [{
        'states': [
            0, 0, 4, 0, 4, 8, 4, 4, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 4, 4, 8, 9,
            8, 8, 9, 10, 14, 15
        ],
        'actions': [
            0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 1,
            3, 3, 1, 0, 1
        ],
        'rewards': [
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            1.0
        ],
        'true_rewards': [],
        'features': [
            np.array([
                1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.
            ]),
            np.array([
                0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.
            ]),
            np.array([
                1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.
            ]),
            np.array([
                0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.
            ]),
            np.array([
                0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.
            ]),
            np.array([
                0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.
            ]),
            np.array([
                0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.
            ]),
            np.array([
                1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.
            ]),
            np.array([
                1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.
            ]),
            np.array([
                1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.
            ]),
            np.array([
                1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.
            ]),
            np.array([
                0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.
            ]),
            np.array([
                1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.
            ]),
            np.array([
                1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.
            ]),
            np.array([
                1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.
            ]),
            np.array([
                1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.
            ]),
            np.array([
                1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.
            ]),
            np.array([
                0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.
            ]),
            np.array([
                0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.
            ]),
            np.array([
                0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.
            ]),
            np.array([
                0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.
            ]),
            np.array([
                0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.
            ]),
            np.array([
                0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.
            ]),
            np.array([
                0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.
            ]),
            np.array([
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.
            ]),
            np.array([
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.
            ]),
            np.array([
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.
            ])
        ]
    },
                    {
                        'states': [0, 4, 8, 8, 9, 10, 6, 2, 6, 10, 14, 15],
                        'actions': [0, 0, 3, 3, 1, 0, 2, 0, 2, 0, 1],
                        'rewards': [
                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                            1.0
                        ],
                        'true_rewards': [],
                        'features': [
                            np.array([
                                0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
                                0., 0., 0., 0.
                            ]),
                            np.array([
                                0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
                                0., 0., 0., 0.
                            ]),
                            np.array([
                                0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
                                0., 0., 0., 0.
                            ]),
                            np.array([
                                0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
                                0., 0., 0., 0.
                            ]),
                            np.array([
                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
                                0., 0., 0., 0.
                            ]),
                            np.array([
                                0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
                                0., 0., 0., 0.
                            ]),
                            np.array([
                                0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                0., 0., 0., 0.
                            ]),
                            np.array([
                                0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
                                0., 0., 0., 0.
                            ]),
                            np.array([
                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
                                0., 0., 0., 0.
                            ]),
                            np.array([
                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                0., 0., 1., 0.
                            ]),
                            np.array([
                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                0., 0., 0., 1.
                            ])
                        ]
                    }]
    metrics = []
    alg = alg_class(env, expert_trajs, rl_alg_factory, metrics, config)
    alg.train(2, 2, 2)


def test_appr_irl_runs():
    quick_run_alg(ApprIRL, {'mode': 'svm'})
    quick_run_alg(ApprIRL, {'mode': 'projection'})


def test_me_irl_runs():
    quick_run_alg(MaxEntIRL)


def test_mce_irl_runs():
    quick_run_alg(MaxCausalEntIRL)
