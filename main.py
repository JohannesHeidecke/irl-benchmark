from irl_benchmark.experiment.run import Run
from irl_benchmark.irl.algorithms.appr_irl import ApprIRL
from irl_benchmark.irl.algorithms.mce_irl import MaxCausalEntIRL
from irl_benchmark.irl.algorithms.me_irl import MaxEntIRL
from irl_benchmark.irl.reward.reward_function import FeatureBasedRewardFunction
from irl_benchmark.metrics.avg_traj_return import AverageTrajectoryReturn
from irl_benchmark.metrics.feature_count_l2 import FeatureCount2Loss
from irl_benchmark.metrics.feature_count_inf import FeatureCountInfLoss
from irl_benchmark.rl.algorithms import ValueIteration


env_id = 'FrozenLake8x8-v0'

expert_trajs_path = 'data/frozen8/expert'

metrics = [AverageTrajectoryReturn, FeatureCount2Loss, FeatureCountInfLoss]

rl_config = {'gamma': 0.9}

# irl_config = {'gamma': 0.9}
irl_config = {}

run_config = {
    'reward_function': FeatureBasedRewardFunction,
    'no_expert_trajs': 10000,
    'no_irl_iterations': 2,
    'no_rl_episodes_per_irl_iteration': 1000,
    'no_irl_episodes_per_irl_iteration': 5000,
}

irl_algs = [MaxCausalEntIRL, MaxEntIRL, ApprIRL]

for irl_alg in irl_algs:

    print('RUNNING: ' + str(irl_alg))

    # factory creating new IRL algorithms:
    def irl_alg_factory(env, expert_trajs, metrics, rl_config, irl_config):
        # factory defining which RL algorithm is used:
        def rl_alg_factory(env):
            return ValueIteration(env, rl_config)

        return irl_alg(env, expert_trajs, rl_alg_factory, metrics, irl_config)

    run = Run(env_id, expert_trajs_path, irl_alg_factory, metrics,
              rl_config, irl_config, run_config)

    run.start()
