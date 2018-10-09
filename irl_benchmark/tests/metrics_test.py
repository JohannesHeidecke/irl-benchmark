import numpy as np
import scipy.optimize
import torch

from irl_benchmark.irl.reward import truth
from irl_benchmark.irl.reward.reward_function import TabularRewardFunction
from irl_benchmark.metrics.reward_l2_loss import RewardL2Loss

float_type = torch.float64
device = torch.device('cpu')


def tabular_l2_loss(x, R_true, R_learned, gamma):
    x = torch.from_numpy(x)
    x.requires_grad_()

    c = x[0]**2  # Prevent negativity
    phi = x[1:]

    R_true, R_learned = map(
        lambda a: a.reshape((a.shape + (1, 1))[:3]),
        [R_true, R_learned])

    t = R_true - c*R_learned - phi.reshape([-1, 1, 1]) + gamma*phi
    loss = torch.sum(t**2)
    loss.backward()
    return loss.detach().numpy(), x.grad.numpy()


def test_l2_metric_s():
    '''
    Tests the L2-minimisation by potentials metric when the reward is only a
    function of the state
    '''
    gamma = 0.9

    rew_true = truth.make('FrozenLake-v0')
    domain = rew_true.domain()
    R_true = torch.from_numpy(rew_true.reward(domain))
    assert R_true.shape == (16,), "16 states"

    env_learned = TabularRewardFunction(
        rew_true.env,
        torch.randn(R_true.shape, device=device, dtype=float_type))
    R_learned = torch.from_numpy(env_learned.reward(domain))

    opt_result = scipy.optimize.minimize(
        fun=tabular_l2_loss, x0=np.ones([17]), args=(R_true, R_learned, gamma),
        method='BFGS', jac=True, options=dict(maxiter=1000), tol=1e-8)
    opt_loss = opt_result.fun
    opt_c = opt_result.x[0]**2
    opt_phi = opt_result.x[1:]

    metric = RewardL2Loss(rew_true)
    loss = metric.evaluate(env_learned, gamma, use_sampling=False)
    c, phi = metric._last_c, metric._last_phi

    assert np.allclose(opt_loss, loss)
    assert np.allclose(opt_c, c)
    assert np.allclose(opt_phi, phi)

    gamma = 0.9

    rew_true = truth.make('FrozenLake8x8-v0')
    domain = rew_true.domain()
    R_true = torch.from_numpy(rew_true.reward(domain))
    assert R_true.shape == (64,), "16 states"

    env_learned = TabularRewardFunction(
        rew_true.env,
        torch.randn(R_true.shape, device=device, dtype=float_type))
    R_learned = torch.from_numpy(env_learned.reward(domain))

    opt_result = scipy.optimize.minimize(
        fun=tabular_l2_loss, x0=np.ones([65]), args=(R_true, R_learned, gamma),
        method='BFGS', jac=True, options=dict(maxiter=1000), tol=1e-8)
    opt_loss = opt_result.fun
    opt_c = opt_result.x[0]**2
    opt_phi = opt_result.x[1:]

    metric = RewardL2Loss(rew_true)
    loss = metric.evaluate(env_learned, gamma, use_sampling=False)
    c, phi = metric._last_c, metric._last_phi

    assert np.allclose(opt_loss, loss)
    assert np.allclose(opt_c, c)
    assert np.allclose(opt_phi, phi)
