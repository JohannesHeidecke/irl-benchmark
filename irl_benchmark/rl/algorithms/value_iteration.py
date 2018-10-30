"""Module for value iteration RL algorithm."""

import gym
import numpy as np

from irl_benchmark.config import RL_CONFIG_DOMAINS
from irl_benchmark.rl.algorithms.base_algorithm import BaseRLAlgorithm
from irl_benchmark.utils.wrapper import is_unwrappable_to, \
    get_transition_matrix, get_reward_matrix


class ValueIteration(BaseRLAlgorithm):
    """Value iteration algorithm.

    Solves an MDP given exact knowledge of transition dynamics and rewards.
    Currently only implemented for DiscreteEnv environments.
    """

    def __init__(self, env: gym.Env, config: dict):
        """

        Parameters
        ----------
        env: gym.Env
            A DiscreteEnv environment
        config: dict
            Configuration of hyperparameters.
        """
        assert is_unwrappable_to(env, gym.envs.toy_text.discrete.DiscreteEnv)
        super(ValueIteration, self).__init__(env, config)
        self.no_states = env.observation_space.n + 1  # + 1 for absorbing state
        self.no_actions = env.action_space.n
        self.transitions = get_transition_matrix(env)
        # will be filled in beginning of training:
        self.rewards = None
        # will be filled during training:
        self.state_values = None
        self.q_values = None
        # whenever self._policy is None, it will be re-calculated
        # based on current self.q_values when calling policy().
        self._policy = None

    def train(self, no_episodes: int):
        """ Train the agent

        Parameters
        ----------
        no_episodes: int
            Not used in this algorithm (since it assumes known transition dynamics)
        """
        assert is_unwrappable_to(self.env,
                                 gym.envs.toy_text.discrete.DiscreteEnv)
        # extract reward function from env (using wrapped reward function if available):
        self.rewards = get_reward_matrix(self.env)

        # initialize state values:
        state_values = np.zeros([self.no_states])

        while True:  # stops when state values converge
            # remember old values for error computation
            old_state_values = state_values.copy()
            # calculate Q-values:
            q_values = self.rewards + \
                       self.config['gamma'] * self.transitions.dot(state_values)
            # calculate state values either with maximum or mellow maximum:
            if self.config['temperature'] is None:
                # using default maximum operator:
                state_values = self._argmax_state_values(q_values)
            else:
                # using softmax:
                state_values = self._softmax_state_values(q_values)

            # stopping condition:
            # check if state values converged (almost no change since last iteration:
            if np.allclose(
                    state_values, old_state_values,
                    atol=self.config['epsilon']):
                break

        # persist learned state values and Q-values:
        self.state_values = state_values
        self.q_values = q_values
        # flag to tell other methods that policy needs to be updated based on new values:
        self._policy = None

    def pick_action(self, state: int) -> int:
        """ Pick an action given a state.

        The way of picking is either uniformly random between all best options (argmax)
        or according to mellowmax distribution, if 'temperature' is not None in config.
        See :meth:`.policy`.

        Parameters
        ----------
        state: int
            An integer corresponding to a state of a DiscreteEnv.

        Returns
        -------
        int
            An action for a DiscreteEnv.
        """
        return np.random.choice(self.no_actions, p=self.policy(state))

    def policy(self, state: int) -> np.ndarray:
        """ Return the probabilities of picking all possible actions given a state.

        The probabilities are either uniformly random between all best options (argmax)
        or according to mellowmax distribution, if 'temperature' is not None in config.

        Parameters
        ----------
        state: int
            An integer corresponding to a state of a DiscreteEnv.

        Returns
        -------
        np.ndarray
            Action probabilities given the state.

        """
        assert self.q_values is not None, "Call train() before calling this method."
        assert np.isscalar(state)
        assert isinstance(state, (int, np.int64))

        return self.policy_array()[state, :]

    def policy_array(self):
        """Return action probabilities for all states as a numpy array.

        Returns
        -------
        np.ndarray
            Array containing probabilities for actions given a state.
            Shape: (n_states, n_actions)
        """
        self._update_policy_if_necessary()
        return self._policy

    def _update_policy_if_necessary(self):
        if self._policy is None:
            if self.config['temperature'] is None:
                self._policy = self._argmax_policy(self.q_values)
            else:
                self._policy = self._softmax_policy(self.q_values)

    def _argmax_policy(self, q_values: np.ndarray) -> np.ndarray:
        """ Calculate an argmax policy.

        Only picks actions with maximal Q-value given a state. If several actions
        are maxima, they are equally likely to be picked.

        Parameters
        ----------
        q_values:
            Q-values for all state-action pairs. Shape (n_states, n_actions)
        Returns
        -------
        np.ndarray
            Probabilities for actions given a state. Shape (n_states, n_actions)
        """
        # Find best actions:
        best_actions = np.isclose(q_values,
                                  np.max(q_values, axis=1).reshape((-1, 1)))
        # Initialize probabilities to be zero
        policy = np.zeros((self.no_states, self.no_actions))
        # Assign probability max to all best actions:
        policy[best_actions] = 1
        # Normalize values so their sum is 1. for each state:
        policy /= np.sum(policy, axis=1, keepdims=True)
        return policy

    def _softmax_policy(self, q_values):
        assert self.config['temperature'] is not None
        temperature = self.config['temperature']
        # for numerical stability (avoiding under- or overflow of exponent),
        # re-scale exponent without changing results of softmax,
        # using softmax(x) = softmax(x + c) for any constant c
        q_max = q_values.max(axis=1, keepdims=True)
        q_scaled = (q_values - q_max) / temperature
        # calculate softmax policy:
        policy = np.exp(q_scaled)
        # normalize values so their sum is 1. for each state:
        policy /= np.sum(policy, axis=1, keepdims=True)
        return policy

    def _mellowmax_policy(self, q_values):
        raise NotImplementedError()

    def _argmax_state_values(self, q_values):
        assert self.config['temperature'] is None
        return np.max(q_values, axis=1)

    def _softmax_state_values(self, q_values):
        assert self.config['temperature'] is not None
        # obtain probabilities of picking each (s, a):
        softmax_policy = self._softmax_policy(q_values)
        # multiply q_values by probability of picking them
        # then sum over actions to get state values:
        softmax_state_values = (softmax_policy * q_values).sum(axis=1)
        return softmax_state_values

    def _mellowmax_state_values(self, q_values):
        raise NotImplementedError()


RL_CONFIG_DOMAINS[ValueIteration] = {
    'gamma': {
        'type': float,
        'min': 0.0,
        'max': 1.0,
        'default': 0.9,
    },
    'epsilon': {
        'type': float,
        'min': 0.0,
        'max': float('inf'),
        'default': 1e-6,
    },
    'temperature': {
        'type': float,
        'optional': True,  # allows value to be None
        'min': 1e-10,
        'max': float('inf'),
        'default': None
    }
}
