import numpy as np
import pickle

from irl_benchmark.config import _irl_algs
from irl_benchmark.irl.algorithms import ApprIRL, MaxEnt
from irl_benchmark.irl.collect import collect_trajs
from irl_benchmark.irl.feature import feature_wrapper
from irl_benchmark.irl.reward import truth
from irl_benchmark.irl.reward.reward_function import  FeatureBasedRewardFunction, TabularRewardFunction
from irl_benchmark.irl.reward.reward_wrapper import RewardWrapper
from irl_benchmark.metrics.inverse_learning_error import ILE
from irl_benchmark.metrics.reward_l2_loss import RewardL2Loss
from irl_benchmark.rl.algorithms.value_iteration import ValueIteration
from irl_benchmark.rl.algorithms import TabularQ
from irl_benchmark.utils.utils import avg_undiscounted_return, get_transition_matrix

TRAIN_FOR = 120

def generate_expert_data():
    def train_and_store(env_id, path):
        env = feature_wrapper.make(env_id)
        expert = ValueIteration(env, gamma=0.8, error=1e-8)
        expert.train(600)
        collect_trajs(env, expert, int(1e4), None, path)
    train_and_store('FrozenLake-v0', 'data/frozen/expert/')
    train_and_store('FrozenLake8x8-v0', 'data/frozen8_8/expert/')


# if you don't have collected expert data at data/frozen/expert
# or data/frozen_8/expert, run the following line.
# the data should be about 124MB and 283MB respectively.
# generate_expert_data()

def rl_alg_factory(env):
    return ValueIteration(env, error=1e-5, gamma=0.8)
    # return TabularQ(env, gamma=0.8)

def run(env_id, agent_id, no_expert_trajs):

    load_from = 'data/frozen/expert/trajs.pkl' if env_id == 'FrozenLake-v0' else 'data/frozen8_8/expert/trajs.pkl'
    with open(load_from, 'rb') as f:
        expert_trajs = pickle.load(f)

    env = feature_wrapper.make(env_id)
    env_size = env.feature_shape()
    reward_function = FeatureBasedRewardFunction(env, np.zeros(env_size))
    env = RewardWrapper(env, reward_function)

    usable_trajs = expert_trajs[:no_expert_trajs]

    if agent_id == 'ApprIRL-Proj':
        irl_alg = ApprIRL(env, usable_trajs, rl_alg_factory, gamma=0.8, proj=True)
        irl_alg.train(TRAIN_FOR)
    elif agent_id == 'MaxEntIRL':
        irl_alg = MaxEnt(env,  expert_trajs=usable_trajs,
                         transition_dynamics=get_transition_matrix(env, with_absorbing_state=False),
                         rl_alg_factory=rl_alg_factory)
        irl_alg.train(feat_map=np.eye(env_size[0]), time_limit=TRAIN_FOR)
    else:
        irl_alg = _irl_algs[agent_id]['alg'](env, usable_trajs,
                                        rl_alg_factory, gamma=0.8)
        irl_alg.train(TRAIN_FOR)

    true_reward_function = truth.make(env_id)
    estim_reward_function = TabularRewardFunction(env, irl_alg.reward_function.parameters)

    ile = ILE(env, true_reward_function, gamma=0.8)
    l2loss = RewardL2Loss(true_reward_function)

    results = {'ile': ile.evaluate(estim_reward_function),
               'l2_loss':l2loss.evaluate(estim_reward_function, gamma=0.8).item()}

    estim_env = feature_wrapper.make(env_id)
    estim_env = RewardWrapper(estim_env, estim_reward_function)
    test_agent = rl_alg_factory(env)
    test_agent.train(60)
    test_trajs = collect_trajs(estim_env, test_agent, 1000, None, None)
    results['avg_return'] = avg_undiscounted_return(test_trajs)

    return results

#
# for env_id in ['FrozenLake-v0', 'FrozenLake8x8-v0']:
#     for agent_id in ['ApprIRL-SVM', 'ApprIRL-Proj', 'MaxEntIRL', 'RelEntIRL']:#, 'MaxCausIRL']:
#         print('= ' * 10)
#         print(agent_id)
#         results = run(env_id, agent_id, 1000)
#         print(results)

