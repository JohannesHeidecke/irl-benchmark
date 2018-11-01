"""This module is used to collect trajectories for inverse reinforcement learning"""

import os
from typing import Dict, List, Union

import gym
import msgpack
import msgpack_numpy as m
from tqdm import tqdm

from irl_benchmark.rl.algorithms.base_algorithm import BaseRLAlgorithm

# set up MessagePack to be numpy compatible:
m.patch()


def collect_trajs(env: gym.Env,
                  agent: BaseRLAlgorithm,
                  no_trajectories: int,
                  max_steps_per_episode: Union[int, None] = None,
                  store_to: Union[str, None] = None,
                  verbose: bool = False) -> List[Dict[str, list]]:
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
        Maximum number of steps allowed to take in each episode. Optional.
        If not set, the environment's default is used.
    store_to: Union[str, None]
        If not None: a path of where trajectories should be persisted.
    verbose: bool
        Whether to use a tqdm progress bar while collecting.

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

    for _ in (tqdm(range(no_trajectories))
              if verbose else range(no_trajectories)):
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
        store_trajs(trajectories, store_to)

    return trajectories


def store_trajs(trajs, store_to):
    """Store trajectories to store_to/trajs.data."""
    if not os.path.exists(store_to):
        os.makedirs(store_to)
    file_path = os.path.join(store_to, 'trajs.data')
    with open(file_path, 'wb') as file:
        msgpack.pack(trajs, file)


def load_stored_trajs(load_from):
    """Return trajectories storead at load_from/trajs.data."""
    file_path = os.path.join(load_from, 'trajs.data')
    with open(file_path, 'rb') as file:
        trajs = msgpack.unpack(file)
    # convert byte keys back to string keys:
    trajs = [{
        key.decode('utf-8') if isinstance(key, bytes) else key: traj[key]
        for key in traj.keys()
    } for traj in trajs]
    return trajs
