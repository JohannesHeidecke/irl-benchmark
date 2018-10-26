from irl_benchmark.envs import envs_feature_based, envs_known_transitions
from irl_benchmark.irl.algorithms import ApprIRL, MaxEnt, RelEnt
from irl_benchmark.irl.reward.reward_function import FeatureBasedRewardFunction

_irl_algs = {
    'ApprIRL-Proj': {
        'alg': ApprIRL,
        'envs': set.intersection(envs_feature_based()),
        'rew_func_type': FeatureBasedRewardFunction
    },
    'ApprIRL-SVM': {
        'alg': ApprIRL,
        'envs': set.intersection(envs_feature_based()),
        'rew_func_type': FeatureBasedRewardFunction
    },
    'MaxEntIRL': {
        'alg': MaxEnt,
        'envs': set.intersection(envs_feature_based(), envs_known_transitions()),
        'rew_func_type': FeatureBasedRewardFunction
    },
    'RelEntIRL': {
        'alg': RelEnt,
        'envs': set.intersection(envs_feature_based()),
        'rew_func_type': FeatureBasedRewardFunction
    }
}



