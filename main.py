from irl_benchmark.experiment.run import Run
from irl_benchmark.irl.algorithms.appr_irl import ApprIRL
from irl_benchmark.irl.algorithms.me_irl import MaxEntIRL
from irl_benchmark.irl.reward.reward_function import FeatureBasedRewardFunction
from irl_benchmark.rl.algorithms import ValueIteration


env_id = 'FrozenLake-v0'

expert_trajs_path = 'data/frozen/expert'

# factory creating new IRL algorithms:
def irl_alg_factory(env, expert_trajs):
    # factory defining which RL algorithm is used:
    def rl_alg_factory(env):
        return ValueIteration(env, {'gamma': 0.9})
    return MaxEntIRL(env, expert_trajs, rl_alg_factory, {'gamma': 0.9})


metrics = []

run_config = {
    'reward_function': FeatureBasedRewardFunction,
    'no_expert_trajs': 5000,
    'no_irl_iterations': 100,
    'no_rl_episodes_per_irl_iteration': 1000,
    'no_irl_episodes_per_irl_iteration': 5000,
}


run = Run(env_id, expert_trajs_path, irl_alg_factory, metrics,
                 run_config)

run.start()