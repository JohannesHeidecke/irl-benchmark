"""Module for the maze world game environment."""
from typing import List, Tuple

import gym
from gym.spaces.multi_binary import MultiBinary
from gym.spaces.discrete import Discrete
import numpy as np
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
from pathfinding.core.diagonal_movement import DiagonalMovement

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# DEFINING CONSTANTS HERE:
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# initial position of the player (7th reward field, index 6)
INIT_POSITION = 6

# define characters for map:
CHAR_WALL = '#'
CHARS_REWARD = ['2', '4', '8']

# define rewards
REWARD_MOVE = -.1
REWARD_SMALL = int(CHARS_REWARD[0])
REWARD_MEDIUM = int(CHARS_REWARD[1])
REWARD_LARGE = int(CHARS_REWARD[2])
# (rewards for reaching fields with CHARS_REWARDS will be 2, 4, 8 respectively)

# define random quit chance
RANDOM_QUIT_CHANCE = 0.02

# define first map
MAP0 = [
    '####################', '#        #   #     #', '# ## ## 2  # # ### #',
    '#  # #### ## # ###4#', '## #4     ##    ## #', '## #### #  #### #  #',
    '#    ## ##  2     ##', '# ## #  #  ## # #  #', '# ## # ## ### #  # #',
    '#  #   ##     ## # #', '## ### ### ##    # #', '##   # 2 # #####   #',
    '#  #   #        4# #', '# #######  ####### #', '#   ######   ##   2#',
    '### # 2##### #  ## #', '#4  ##  #### ## ## #', '# # ###         ####',
    '###     #######   4#', '####################'
]

# define second map (2s and 4s swapped)
MAP1 = [
    '####################', '#        #   #     #', '# ## ## 4  # # ### #',
    '#  # #### ## # ###2#', '## #2     ##    ## #', '## #### #  #### #  #',
    '#    ## ##  4     ##', '# ## #  #  ## # #  #', '# ## # ## ### #  # #',
    '#  #   ##     ## # #', '## ### ### ##    # #', '##   # 4 # #####   #',
    '#  #   #        2# #', '# #######  ####### #', '#   ######   ##   4#',
    '### # 4##### #  ## #', '#2  ##  #### ## ## #', '# # ###         ####',
    '###     #######   2#', '####################'
]

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# END OF CONSTANTS DEFINITION
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def get_maps(ascii_map: List[str]) -> (np.ndarray, np.ndarray):
    """Transform an ASCII-ascii_map into matrices of walls and rewards

    Parameters
    ----------
    ascii_map: : List[str]
        A list of strings where each entry describes one row of the ascii_map.

    Returns
    -------
    (np.ndarray, np.ndarray)
        (walls, rewards) with shapes (len(MAP), len(MAP[0])).
        Walls will be indicated by the value 1.0 in the first returned array.
        Rewards will be indicated by non-zero values according to their value.
    """
    map_walls = np.zeros((len(ascii_map), len(ascii_map[0])))
    map_rewards = np.zeros((len(ascii_map), len(ascii_map[0])))
    for r, row in enumerate(ascii_map):
        for c, char in enumerate(row):
            if char == CHAR_WALL:
                map_walls[r, c] = 1.0
            elif char in CHARS_REWARD:
                map_rewards[r, c] = float(char)
    return map_walls, map_rewards


def get_rew_coords(map_rewards: np.ndarray) -> List[Tuple[int, int]]:
    """Get coordinates for all rewards in map_rewards.

    Parameters
    ----------
    map_rewards: np.ndarray
        An array with non-zero values wherever there is a reward field.
        Should be in the format as outputted from :func:`.get_maps`.

    Returns
    -------
    List[Tuple[int, int]]
        A list containing 2-tuples, where each 2-tuple contains a coordinate
        of one reward field. Ordered top-left to bottom-right.
    """
    rews_where = np.where(map_rewards > 0)
    rew_coords = []
    for i in range(len(rews_where[0])):
        rew_coords.append((int(rews_where[0][i]), int(rews_where[1][i])))
    return rew_coords


def new_init_state(pos_index: int, num_rewards: int) -> np.ndarray:
    """Produce a new initial state where player is at specified initial position.

    Parameters
    ----------
    pos_index: int
        The index of the initial position. Player will be placed at the
        (i+1)th reward field and the reward at this field will be marked as collected.
    num_rewards: int
        The number of rewards on the map. The returned state will be of shape
        (2 * num_rewards,).

    Returns
    -------
    np.ndarray
        An initial state. Shape (2 * num_rewards,). The first num_rewards fields encode
        the player's position, zeros everywhere except the position of the player which
        will be 1.0. The last num_rewards fields encode what rewards are still available.
        All of these last values will be 1.0 (available) except for the starting field,
        which will be 0.0.
    """
    state_pos = np.zeros(num_rewards)
    state_rew = np.ones(num_rewards)
    state_pos[pos_index] = 1.0
    state_rew[pos_index] = 0.0
    state = np.concatenate((state_pos, state_rew), axis=0)
    return state


class MazeWorld(gym.Env):
    """The maze world game class.

    The game consists of a maze with walls with reward cells distributed over the map.
    The player can walk to certain reward cells (number of actions == number of reward cells),
    but on each step there is a small percentage of the game terminating.
    The objective of the game is to collect the rewards in the most efficient way, i.e. walking
    short distances and collecting big rewards as early as possible.
    """

    def __init__(self, map_id: int = 0):
        """

        Parameters
        ----------
        map_id: int
            Which version of the game to play. Legal values are 0 and 1. Default is 0.
            Corresponds to the hard-coded maps MAP0 and MAP1 defined in the same module.
        """
        if map_id == 0:
            used_map = MAP0
        elif map_id == 1:
            used_map = MAP1
        # if adding more values here, also adapt docstring above.
        else:
            raise NotImplementedError()

        # get arrays of walls and reward fields:
        self.map_walls, self.map_rewards = get_maps(used_map)
        self.rews_where = np.where(self.map_rewards > 0)
        # get list of coordinates for reward fields:
        self.rew_coords = get_rew_coords(self.map_rewards)
        # get number of rewards in map_id:
        self.num_rewards = len(self.rews_where[0])

        # calculate all possible paths, using the 'pathfinding' library.
        self.paths = {}
        matrix = np.swapaxes(np.abs(self.map_walls - 1.0), 0, 1).tolist()
        finder = AStarFinder(diagonal_movement=DiagonalMovement.never)
        for i in range(self.num_rewards):
            self.paths[i] = {}
            for j in range(self.num_rewards):
                grid = Grid(matrix=matrix)
                start = grid.node(self.rews_where[0][i], self.rews_where[1][i])
                end = grid.node(self.rews_where[0][j], self.rews_where[1][j])
                path, _ = finder.find_path(start, end, grid)
                path = [[int(x[0]), int(x[1])] for x in path]
                self.paths[i][j] = path

        super(MazeWorld, self).__init__()

        # set observation space and action space:
        self.observation_space = MultiBinary(self.num_rewards * 2)
        self.action_space = Discrete(self.num_rewards)

        self.current_state = None
        self.terminated = True

    def reset(self):
        """Reset the game and return new initial state."""
        # hard coded initial position index needs to be
        # lower than number of reward fields:
        assert self.num_rewards > INIT_POSITION
        start_pos = INIT_POSITION
        # get new initial state:
        self.current_state = new_init_state(start_pos, self.num_rewards)
        # game is not terminated yet:
        self.terminated = False
        # return a copy, so internal state can't be manipulated:
        return np.copy(self.current_state)

    def step(self, action):
        """Do a step with the provided action."""
        state = np.copy(self.current_state)
        assert len(state) == self.num_rewards * 2
        assert action < self.num_rewards
        assert np.sum(
            state[:self.num_rewards]) == 1.0, "Game already terminated?"

        if self.terminated:
            print('CALLED STEP EVEN THOUGH ALREADY TERMINATED!! :(')
            self.current_state = state
            return state, 0, True, {}

        pos_index = int(np.where(state[:self.num_rewards] > 0)[0][0])

        reward = 0

        state[:self.num_rewards] = 0.0

        if pos_index == action:
            if RANDOM_QUIT_CHANCE > np.random.uniform():
                self.terminated = True
            else:
                state[action] = 1.
            self.current_state = state
            return state, REWARD_MOVE, self.terminated, {}

        path = self.paths[pos_index][action]

        for _ in [path[i] for i in range(1, len(path))]:
            if RANDOM_QUIT_CHANCE > np.random.uniform():
                self.terminated = True
                break
            reward += REWARD_MOVE

        if not self.terminated:
            state[action] = 1.0

            reward_value = self.get_rew_value(state, action)

            if np.isclose(state[self.num_rewards + action], 1.):
                state[self.num_rewards + action] = 0.0
                if sum(state[self.num_rewards:]) == 0:
                    self.terminated = True

            reward += reward_value
        self.current_state = state
        return state, reward, self.terminated, {}

    def get_paths(self) -> dict:
        """Return paths between reward fields as a dictionary."""
        return self.paths

    def get_path_len(self, state, action):
        """Return the length of the path when being in
        state 'state' and performing action 'action'."""
        pos_index = int(np.where(state[:self.num_rewards] > 0)[0][0])
        path = self.paths[pos_index][action]
        return len(path)

    def get_rew_value(self, state, action):
        """Return the reward value of being in state 'state' and
        reaching the next state after performing 'action' without
        a random termination event."""
        if np.isclose(state[self.num_rewards + action], 1.):
            rew_value = float(
                self.map_rewards[self.rews_where[0][action], self.
                                 rews_where[1][action]])
        else:
            rew_value = 0.
        return rew_value

    def index_to_state(self, index: int) -> np.ndarray:
        """Return the index of a given state.

        Parameters
        ----------
        index: int
            A state's index value.

        Returns
        -------
        np.ndarray
            The state variable vector corresponding to the given index.
        """
        num_rewards = self.action_space.n
        pos_index = int(index / (2**num_rewards))
        state_pos = np.zeros(num_rewards)
        state_pos[pos_index] = 1.0
        state_rew = np.zeros(num_rewards)
        for i in range(num_rewards):
            state_rew[num_rewards - 1 - i] = index % 2
            index = int(index / 2)
        state = np.concatenate((state_pos, state_rew), axis=0)
        return state

    def state_to_index(self, state: np.ndarray) -> int:
        """Return an index value for a state.

        Parameters
        ----------
        state: np.ndarray
            The state for which to return its index.

        Returns
        -------
        int:
            The index of the given state.
        """
        num_rewards = self.action_space.n
        position = np.where(state[:num_rewards] > 0)
        if position[0].size == 0:
            # return terminal state
            return num_rewards * 2**num_rewards
        pos_index = int(position[0][0])
        rew_dec = 0
        rew_part = state[num_rewards:]
        for i in range(num_rewards):
            rew_dec += rew_part[num_rewards - 1 - i] * 2**i
        index = pos_index * (2**num_rewards) + rew_dec
        return int(index)

    def render(self, mode='human'):
        """Not implemented for MazeWorld."""
        raise NotImplementedError()
