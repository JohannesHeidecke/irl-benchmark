import numpy as np
from typing import Union

State = namedtuple('State', ('state'))
StateAction = namedtuple('StateAction', ('state', 'action'))
StateActionState = namedtuple('StateActionState', ('state', 'action', 'next_state'))

class RewardFunction(object):

    def __init__(self):
        raise NotImplementedError()

    def domain(self) -> Union[State, StateAction, StateActionState]:
        raise NotImplementedError()

    def domain_sampler(self, batch_size) -> Union[State, StateAction, StateActionState]:
        raise NotImplementedError()

    def reward(self, input) -> np.ndarray:
        raise NotImplementedError()


class RewardFunctionS(RewardFunction):
    def reward(self, input: State) -> np.ndarray:
        # input: batch of states
        # returns: np.ndarray, (n)


class RewardFunctionSa(RewardFunction):
    def reward(self, input: StateAction) -> np.ndarray:
        # input: batch of state action pairs
        # returns: np.ndarray, (n)


class RewardFuncitonSas(RewardFunction):
    def reward(self, input: StateActionState) -> np.ndarray:
        # input: batch of state action state triplets
        # returns: np.ndarray, (n)