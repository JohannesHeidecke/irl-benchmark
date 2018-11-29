from irl_benchmark.config import IRL_ALG_REQUIREMENTS, RL_ALG_REQUIREMENTS
from irl_benchmark.envs import envs_feature_based, envs_known_transitions
from irl_benchmark.experiment.run import Run
from irl_benchmark.irl.algorithms.appr_irl import ApprIRL
from irl_benchmark.irl.algorithms.mce_irl import MaxCausalEntIRL
from irl_benchmark.irl.algorithms.me_irl import MaxEntIRL
from irl_benchmark.irl.reward.reward_function import FeatureBasedRewardFunction
from irl_benchmark.metrics.avg_traj_return import AverageTrajectoryReturn
from irl_benchmark.metrics.feature_count_l2 import FeatureCount2Loss
from irl_benchmark.metrics.feature_count_inf import FeatureCountInfLoss
from irl_benchmark.rl.algorithms import ValueIteration

env_id = 'MazeWorld0-v0'

expert_trajs_path = 'data/maze0/expert_human'

metrics = [AverageTrajectoryReturn, FeatureCount2Loss, FeatureCountInfLoss]

rl_config = {'gamma': 1.0}  #, 'temperature': .5}

irl_config = {'gamma': 1.0}

run_config = {
    'reward_function': FeatureBasedRewardFunction,
    'no_expert_trajs': 425,
    'no_irl_iterations': 200,
    'no_rl_episodes_per_irl_iteration': 1000,
    'no_irl_episodes_per_irl_iteration': 10000,
    'no_metric_episodes_per_irl_iteration': 10000,
}

irl_algs = [ApprIRL]  #[MaxCausalEntIRL, MaxEntIRL, ApprIRL]

rl_alg = ValueIteration

for irl_alg in irl_algs:
    if IRL_ALG_REQUIREMENTS[irl_alg]['requires_features']:
        assert env_id in envs_feature_based()
    if IRL_ALG_REQUIREMENTS[irl_alg]['requires_transitions']:
        assert env_id in envs_known_transitions()

for irl_alg in irl_algs:

    # factory creating new IRL algorithms:
    def irl_alg_factory(env, expert_trajs, metrics, rl_config, irl_config):
        # factory defining which RL algorithm is used:
        def rl_alg_factory(env):
            return rl_alg(env, rl_config)

        return irl_alg(env, expert_trajs, rl_alg_factory, metrics, irl_config)

    run_config['requires_features'] = IRL_ALG_REQUIREMENTS[irl_alg][
        'requires_features']
    run_config['requires_transitions'] = IRL_ALG_REQUIREMENTS[irl_alg][
        'requires_transitions'] or RL_ALG_REQUIREMENTS[rl_alg][
            'requires_transitions']

    print('RUNNING: ' + str(irl_alg))

    run = Run(env_id, expert_trajs_path, irl_alg_factory, metrics, rl_config,
              irl_config, run_config)

    run.start()
