"""This module is used to collect trajectories for inverse reinforcement learning"""

import os
import pickle
from typing import Dict, List, Union

import gym

from irl_benchmark.rl.algorithms.base_algorithm import BaseRLAlgorithm


def collect_trajs(env: gym.Env,
                  agent: BaseRLAlgorithm,
                  no_trajectories: int,
                  max_steps_per_episode: int,
                  store_to: Union[str, None] = None) -> List[Dict[str, list]]:
    """ Collect and return trajectories of an agent in an environment.

    Parameters
    ----------
    env: gym.Env
        A gym environment
    agent: BaseRLAlgorithm
        An RL algorithm / agent
    no_trajectories: int
        Number of trajectories to be collected
    max_steps_per_episode: int
        Maximum number of steps allowed to take in each episode.
    store_to: Union[str, None]
        If not None: a path of where trajectories should be persisted.

    Returns
    -------
    List[Dict[str, list]]
        A list of trajectories. Each trajectory is a dictionary with keys
        ['states', 'actions', 'rewards', 'true_rewards', 'features']. The
        values of each dictionary are lists. The list of states contains
        both starting and final state and is one element longer than the
        list of actions. The lists of true_rewards and features can be
        empty if the environment was not wrapped in a RewardWrapper or
        FeatureWrapper respectively.
    """

    # list of trajectories to be returned:
    trajectories = []

    for trajectory in range(no_trajectories):
        # start new episode by resetting environment:
        state = env.reset()
        done = False

        # trajectories contain starting state.
        # this makes len(states) to be one larger than len(actions)
        states = [state]
        # initialize other lists:
        actions = []
        rewards = []
        true_rewards = []
        features = []

        # if no max steps are specified,
        # check if there is a default provided by the environment:
        if env.spec.max_episode_steps is not None:
            assert isinstance(env.spec.max_episode_steps, int)
            if max_steps_per_episode is None:
                max_steps_per_episode = env.spec.max_episode_steps
            elif env.spec.max_episode_steps < max_steps_per_episode:
                print(
                    'WARNING: running episodes for longer than the default of '
                    + str(env.spec.max_episode_steps) +
                    ' for this environment.')

        step_counter = 0
        while not done \
            and (max_steps_per_episode is None or step_counter <
                 max_steps_per_episode):
            step_counter += 1
            # Agent picks an action:
            action = agent.pick_action(state)
            # Action is executed by environment:
            next_state, reward, done, info = env.step(action)
            # Append obtained information to respective lists:
            states.append(next_state)
            actions.append(action)
            rewards.append(reward)
            if 'true_reward' in info.keys():
                true_rewards.append(info['true_reward'])
            if 'features' in info.keys():
                features.append(info['features'])
            # Update which state we are in:
            state = next_state

        # Construct trajectory dictionary:
        trajectory = {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'true_rewards': true_rewards,
            'features': features
        }
        # Append trajectory to list of trajectories:
        trajectories.append(trajectory)

    # If requested, store trajetories to file:
    if store_to is not None:
        if not os.path.exists(store_to):
            os.makedirs(store_to)
        with open(store_to + 'trajs.pkl', 'wb+') as file:
            pickle.dump(trajectories, file)

    return trajectories
