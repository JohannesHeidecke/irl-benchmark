"""Module for the reward wrapper, hiding the original reward function and using a different one."""
from typing import Union, Tuple

import gym
import numpy as np

from irl_benchmark.irl.reward.reward_function import State
from irl_benchmark.irl.reward.reward_function import StateAction
from irl_benchmark.irl.reward.reward_function import StateActionState
from irl_benchmark.irl.reward.reward_function import BaseRewardFunction, \
    FeatureBasedRewardFunction, TabularRewardFunction


class RewardWrapper(gym.Wrapper):
    """Use a given reward function instead of the true reward provided by the environment"""

    def __init__(self, env: gym.Env, reward_function: BaseRewardFunction):
        """

        Parameters
        ----------
        env: gym.Env
            A gym environment (potentially already wrapped).
        reward_function: BaseRewardFunction
            The reward function to be used instead of the original reward.
        """
        super(RewardWrapper, self).__init__(env)
        self.reward_function = reward_function
        self.current_state = None

    def reset(self, **kwargs):
        """Call base class reset method and return initial state."""
        # pylint: disable=E0202
        self.current_state = self.env.reset()
        return self.current_state

    def step(self, action: Union[np.ndarray, int, float]
             ) -> Tuple[Union[np.ndarray, float, int], float, bool, dict]:
        """Call base class step method but replace reward with reward output by reward function.

        Parameters
        ----------
        action: Union[np.ndarray, int, float]
            An action, suitable for wrapped environment.

        Returns
        -------
        Tuple[Union[np.ndarray, float, int], float, bool, dict]
            Tuple with values for (state, reward, done, info).
            Normal return values of any gym step function. The reward value is replaced with
            the value output by the reward function given to the wrapper. The true reward is
            added as a field to the dictionary in the last element of the tuple
            with key 'true_reward'.
        """
        # pylint: disable=E0202

        # execute action:
        next_state, reward, terminated, info = self.env.step(action)

        # persist true reward in information:
        info['true_reward'] = reward

        # generate input for reward function:
        if isinstance(self.reward_function, FeatureBasedRewardFunction):
            # reward function is feature based
            rew_input = info['features']
        elif isinstance(self.reward_function, TabularRewardFunction):
            rew_input = self.get_tabular_reward_input(self.current_state,
                                                      action, next_state)
        else:
            raise NotImplementedError()

        reward = self.reward_function.reward(rew_input).item()

        # remember which state we are in:
        self.current_state = next_state

        return next_state, reward, terminated, info

    def get_tabular_reward_input(
            self, state: Union[np.ndarray, int, float],
            action: Union[np.ndarray, int, float],
            next_state: Union[np.ndarray, int, float]
    ) -> Union[State, StateAction, StateActionState]:
        """ Return an adequate input batch for a tabular reward function.

        Parameters
        ----------
        state: state: Union[np.ndarray, int, float]
        action: Union[np.ndarray, int, float]
        next_state: Union[np.ndarray, int, float]

        Returns
        -------
        Union[State, StateAction, StateActionState]
            A domain batch with one element.
        """
        if self.reward_function.action_in_domain:
            if self.reward_function.next_state_in_domain:
                return StateActionState(state, action, next_state)
            return StateAction(state, action)
        if state is None and next_state is not None:
            state = next_state
        return State(state)

    def update_reward_function(self, reward_function):
        """Update the used reward function.

        Useful as IRL algorithms compute a new reward function
        in each iteration."""
        self.reward_function = reward_function

    def get_reward_input_for(self, state: Union[np.ndarray, int, float],
            action: Union[np.ndarray, int, float],
            next_state: Union[np.ndarray, int, float]) -> Union[State, StateAction, StateActionState]:
        """

        Parameters
        ----------
        state: Union[np.ndarray, int, float
        action: Union[np.ndarray, int, float]
        next_state: Union[np.ndarray, int, float]

        Returns
        -------
        Union[State, StateAction, StateActionState]
            The input converted to an adequate namedtuple.
        """
        if self.reward_function.action_in_domain:
            if self.reward_function.next_state_in_domain:
                return StateActionState(state, action, next_state)
            else:
                return StateAction(state, action)
        else:
            if state is None and next_state is not None:
                state = next_state
            return State(state)
