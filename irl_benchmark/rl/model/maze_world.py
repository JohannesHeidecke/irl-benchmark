import numpy as np
import sparse
from tqdm import tqdm

from irl_benchmark.envs.maze_world import MazeWorld, RANDOM_QUIT_CHANCE, REWARD_MOVE
from irl_benchmark.irl.reward.reward_wrapper import RewardWrapper
from irl_benchmark.rl.model.model_wrapper import BaseWorldModelWrapper
from irl_benchmark.utils.wrapper import is_unwrappable_to, unwrap_env


def get_next_state(state: np.ndarray, action: int, num_rewards: int):
    """Return the new state given a state and an action. Assumes the action is
    successful (no random quit chance happened).

    Parameters
    ----------
    state: np.ndarray
        A state. Numpy array of shape (2 * num_rewards,).The first num_rewards entries specify
        the location (one-hot encoded). The last num_rewards entries specify which rewards
        have been collected already.
    action: int
        Index of the reward field the agent walks to.
    num_rewards: int
        How many rewards are on the map.

    Returns
    -------
    np.ndarray
        The next state. The position has changed according to action, and if there was a reward
        at the corresponding field, this field changes to a 0.
    """
    pos_index = action
    state_pos = np.zeros(num_rewards)
    state_pos[pos_index] = 1.0
    state_rew = np.copy(state[num_rewards:])
    state_rew[action] = 0.0
    new_state = np.concatenate((state_pos, state_rew), axis=0)
    return new_state


class MazeModelWrapper(BaseWorldModelWrapper):
    def __init__(self, env):
        assert is_unwrappable_to(env, MazeWorld)
        super(MazeModelWrapper, self).__init__(env)
        self.maze_env = unwrap_env(self.env, MazeWorld)

    def index_to_state(self, index):
        return self.maze_env.index_to_state(index)

    def state_to_index(self, state):
        return self.maze_env.state_to_index(state)

    def n_states(self):
        num_rewards = self.maze_env.num_rewards
        return 2**num_rewards * num_rewards

    def get_transition_array(self):

        return self._get_model_arrays(
            return_transitions=True, return_rewards=False)

    def get_reward_array(self):

        return self._get_model_arrays(
            return_transitions=False, return_rewards=True)

    def _get_model_arrays(self, return_transitions=True, return_rewards=True):

        if return_rewards:
            if is_unwrappable_to(self.env, RewardWrapper):
                reward_wrapper = unwrap_env(self.env, RewardWrapper)
            else:
                reward_wrapper = None

        assert return_transitions or return_rewards

        # +1 for absorbing state:
        n_states = self.n_states() + 1
        absorbing_s = n_states - 1
        num_rewards = n_actions = self.maze_env.action_space.n
        paths = self.maze_env.paths

        if return_transitions:
            coords_trans_state = []
            coords_trans_action = []
            coords_trans_next_state = []
            trans_data = []

            def add_transition(s, a, sn, p):
                coords_trans_state.append(s)
                coords_trans_action.append(a)
                coords_trans_next_state.append(sn)
                trans_data.append(p)

        if return_rewards:
            rewards = np.zeros((n_states, n_actions))

        for s in tqdm(range(n_states - 1)):
            for a in range(n_actions):
                state = self.index_to_state(s)

                if return_rewards and reward_wrapper is not None:
                    rew_input = reward_wrapper.get_reward_input_for(
                        state, a, None)
                    wrapped_reward = reward_wrapper.reward_function.reward(
                        rew_input).item()

                if np.sum(state[num_rewards:]) == 0:
                    if return_transitions:
                        add_transition(s, a, absorbing_s, 1.)
                    if return_rewards:
                        if reward_wrapper is None:
                            rewards[s, a] = 0
                        else:
                            rewards[s, a] = wrapped_reward
                    continue

                pos_index = int(np.where(state[:num_rewards] > 0)[0][0])
                path = paths[pos_index][a]

                if len(path) == 1 or pos_index == a:
                    assert pos_index == a
                    if return_transitions:
                        add_transition(s, a, s, 1. - RANDOM_QUIT_CHANCE)
                        add_transition(s, a, absorbing_s, RANDOM_QUIT_CHANCE)
                    if return_rewards:
                        if reward_wrapper is None:
                            rewards[s, a] = REWARD_MOVE
                            if state[num_rewards + a] != 0:
                                rews_where = self.maze_env.rews_where
                                rewards[s, a] += float(
                                    self.maze_env.map_rewards[rews_where[0][a], \
                                    rews_where[1][a]]) * (1 - RANDOM_QUIT_CHANCE)
                        else:
                            rewards[s, a] = wrapped_reward
                    continue

                success_prob = (1 - RANDOM_QUIT_CHANCE)**(len(path) - 1)
                if return_transitions:
                    new_state = get_next_state(state, a, num_rewards)
                    new_s = self.state_to_index(new_state)
                    add_transition(s, a, new_s, success_prob)
                    add_transition(s, a, absorbing_s, 1. - success_prob)

                if return_rewards:
                    if reward_wrapper is None:
                        if state[num_rewards + a] == 0:
                            # if reward is already collected at this field:
                            rew_value = 0
                        else:
                            rews_where = self.maze_env.rews_where
                            rew_value = float(self.maze_env.map_rewards[
                                rews_where[0][a], rews_where[1][a]])

                        possible_distances = np.arange(1, len(path))
                        prob_getting_to_distance = (
                            1 - RANDOM_QUIT_CHANCE)**possible_distances
                        prob_stopping_at_distance = np.ones_like(
                            possible_distances, dtype=np.float32)
                        prob_stopping_at_distance[:-1] = RANDOM_QUIT_CHANCE
                        expected_walking_distance = np.sum(
                            possible_distances * prob_getting_to_distance *
                            prob_stopping_at_distance)
                        weighted_reward = expected_walking_distance * REWARD_MOVE + success_prob * rew_value

                        rewards[s, a] = weighted_reward
                    else:
                        rewards[s, a] = wrapped_reward

        for a in range(n_actions):
            if return_transitions:
                add_transition(absorbing_s, a, absorbing_s, 1.)
            if return_rewards:
                rewards[absorbing_s, a] = 0

        if return_transitions:
            coords = np.array([
                coords_trans_state, coords_trans_action,
                coords_trans_next_state
            ])
            transitions = sparse.COO(coords, trans_data)

        if return_transitions:
            if return_rewards:
                return transitions, rewards
            return transitions
        return rewards
