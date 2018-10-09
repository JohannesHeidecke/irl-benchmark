import gym

from irl_benchmark.irl.algorithms.lp.LP3 import LP3
from irl_benchmark.rl.algorithm import RandomAgent
from irl_benchmark.rl.algorithms import collect_trajs

# Define important script constants here:
store_to = 'data/frozen/expert'
no_episodes = 1000
max_steps_per_episode = 100

env = gym.make('FrozenLake-v0')
expert_agent = RandomAgent(env)

expert_trajs = collect_trajs(env, expert_agent, no_episodes,
                             max_steps_per_episode, store_to)

lp3 = LP3(env, expert_trajs, RandomAgent)

lp3.train(300)

print(lp3.reward_function())
