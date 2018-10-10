import numpy as np
import torch
from ..irl.reward.reward_function import (
    AbstractRewardFunction, TabularRewardFunction, State)

__all__ = ['RewardL2Loss']


class RewardL2Loss(object):
    '''L2 distance between true reward and closest shaping of estimated reward.

    This is a metric not used in the literature so far.

    Many different reward functions lead to the same ordering of policies.
    It therefore doesn't make sense to look at the L2 distance of individual
    reward functions. Instead, we rely on Ng, Harada & Russell's (1999)
    characterization of the equivalence class of rewards inducing the same
    optimal policy in all MDPs. We search for the reward in the equivalence
    class of the IRL reward output that minimizes the L2 distance to the true
    reward and return that distance.
    '''
    def __init__(self, true_reward: AbstractRewardFunction):
        '''Pass true reward.'''
        self.true_reward = true_reward

    def evaluate(self, estim_reward: AbstractRewardFunction,
                 gamma: float, use_sampling=False, use_gpu=True) -> float:
        '''Return distance between estim_reward and true reward.'''
        if use_gpu and torch.cuda.device_count() > 0 and torch.cuda.is_available():
            device = torch.cuda.device(0)
        else:
            use_gpu = False
            device = torch.device('cpu')

        if use_sampling:
            return NotImplementedError()
        else:
            domain = estim_reward.domain()
            if not isinstance(domain, State):
                raise NotImplementedError(
                    "StateAction or StateActionState rewards")
            R_true = self.true_reward.reward(domain)
            R_est = estim_reward.reward(domain)

            if use_gpu:
                R_true = torch.tensor(R_true, device=device)
                R_est = torch.tensor(R_est, device=device)
            else:
                R_true = torch.from_numpy(R_true)
                R_est = torch.from_numpy(R_est)

            phi_true = _opt_potential(R_true, gamma)
            vija_true = _scalable_term(R_true, gamma, phi_true)
            phi_est = _opt_potential(R_est, gamma)
            vija_est = _scalable_term(R_est, gamma, phi_est)

            c_ = (torch.dot(vija_est, vija_true)
                  / torch.dot(vija_est, vija_est))
            # Only nonnegative scalings c are valid
            self._last_c = c = max(c_.numpy(), 0.0)
            self._last_phi = phi_true - c*phi_est

            diff = vija_true - c*vija_est
            return diff.dot(diff)


def _elements_of_M(gamma, N):
    "Solve linear system for 1st derivative of phi"
    A = (1+gamma**2)*N - 2*gamma
    b = -2*gamma
    return np.linalg.solve([[A, b*(N-1)], [b, A+b*(N-2)]], [[1], [0]])


def _opt_potential(R, gamma):
    "Return the optimal potential function associated with a reward, given M."
    N = R.shape[0]  # number of states
    A = None        # number of actions

    # Get correct Ri_ and R_j depending on shape of R
    if len(R.shape) >= 2:
        A = R.shape[1]

        if len(R.shape) == 3:
            Rij = R.sum(dim=1)
        else:
            Rij = R
        Ri_ = Rij.sum(dim=1)
        R_j = Rij.sum(dim=0)
    else:
        A = 1
        Ri_ = R * N
        R_j = R.sum(dim=0)

    # Elements of matrix M
    diag_M, off_diag_M = _elements_of_M(gamma, N)

    v = (-Ri_ + gamma*R_j)/A
    return -off_diag_M*v.sum() + (off_diag_M - diag_M)*v


def _scalable_term(R, gamma, phi):
    R_full = R.reshape((R.shape + (1, 1))[:3])
    return ((R_full - phi.reshape([-1, 1, 1])
             + gamma*phi.reshape([1, 1, -1]))
            .reshape([-1]))
