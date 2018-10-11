import gym
import logging
import numpy as np

from slm_lab.lib.logger import set_level as set_logging_level

from irl_benchmark.irl.collect import collect_trajs
from irl_benchmark.irl.feature import feature_wrapper
from irl_benchmark.irl.feature.feature_wrapper import PendulumFeatureWrapper
from irl_benchmark.irl.reward.reward_function import FeatureBasedRewardFunction
from irl_benchmark.irl.reward.reward_wrapper import RewardWrapper
from irl_benchmark.rl.algorithms.ppo import PPO

set_logging_level(logging.CRITICAL)

def test_ppo_pendulum():
    env = gym.make('Pendulum-v0')
    agent = PPO(env)
    agent.train(2)
    trajs = collect_trajs(env, agent, 1, None)

def test_ppo_pendulum_wrapped():
    env = feature_wrapper.make('Pendulum-v0')
    agent = PPO(env)
    agent.train(2)
    assert isinstance(agent.agent.body.env.u_env, PendulumFeatureWrapper)
    trajs = collect_trajs(env, agent, 1, None)
    assert np.array(trajs[0]['features']).shape == (200, 3)
    reward_function = FeatureBasedRewardFunction(env, np.zeros(3))
    env = RewardWrapper(env, reward_function)
    agent = PPO(env)
    agent.train(2)
    assert isinstance(agent.agent.body.env.u_env, RewardWrapper)
    trajs = collect_trajs(env, agent, 1, None)
    assert np.array(trajs[0]['true_rewards']).shape == (200,)
    assert np.sum(trajs[0]['rewards']) == 0
    assert np.sum(trajs[0]['true_rewards']) < 0

def test_ppo_lunar():
    env = gym.make('LunarLander-v2')
    agent = PPO(env)
    agent.train(2)
    trajs = collect_trajs(env, agent, 1, None)
