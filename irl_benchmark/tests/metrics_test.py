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


def l2_metric_s(key, states, gamma=0.9):
    '''
    Tests the L2-minimisation by potentials metric when the reward is only a
    function of the state
    '''
    rew_true = truth.make(key)
    domain = rew_true.domain()
    R_true = torch.from_numpy(rew_true.reward(domain))
    assert R_true.shape == (states,), f'{states} states'

    rew_learned = TabularRewardFunction(
        rew_true.env,
        torch.randn(R_true.shape, device=device, dtype=float_type))
    R_learned = torch.from_numpy(rew_learned.reward(domain))

    opt_result = scipy.optimize.minimize(
        fun=tabular_l2_loss, x0=np.ones([states+1]), args=(R_true, R_learned, gamma),
        method='BFGS', jac=True, options=dict(maxiter=1000), tol=1e-8)
    opt_loss = opt_result.fun
    opt_c = opt_result.x[0]**2
    opt_phi = opt_result.x[1:]

    metric = RewardL2Loss(rew_true)
    loss = metric.evaluate(rew_learned, gamma, use_sampling=False)
    c, phi = metric._last_c, metric._last_phi

    assert np.allclose(opt_loss, loss)
    assert np.allclose(opt_c, c)
    assert np.allclose(opt_phi, phi)


def test_l2_metric_s_analytic():
    torch.manual_seed(12)
    for key, states in [
            ('FrozenLake-v0', 16),
            ('FrozenLake8x8-v0', 64)]:
        l2_metric_s(key, states)


def l2_metric_nnet_s(key, states, gamma=0.9):
    '''
    Tests the L2-minimisation by potentials metric when the reward is only a
    function of the state, and we can't handle the full domain. For this
    reason, we parameterise the phi mappings with a neural network.
    '''
    env_true = truth.make(key)
    domain = env_true.domain()
    R_true = torch.from_numpy(env_true.reward(domain))
    assert R_true.shape == (states,), f'{states} states'

    env_learned = TabularRewardFunction(
        env_true.env,
        torch.randn(R_true.shape, device=device, dtype=float_type))

    metric = RewardL2Loss(env_true)
    loss_analytic = float(metric.evaluate(env_learned, gamma, use_sampling=False))
    loss = float(metric.evaluate(env_learned, gamma, use_sampling=True,
                                 n_iters=50000))
    assert np.allclose(loss_analytic, loss, rtol=1e-3)


def test_l2_metric_s_nnet():
    torch.manual_seed(11)
    for key, states in [
            ('FrozenLake-v0', 16),
            ('FrozenLake8x8-v0', 64)]:
        l2_metric_nnet_s(key, states)


def test_l2_basic_properties(gamma=0.9):
    torch.manual_seed(10)
    env_true = truth.make('FrozenLake-v0')
    env_learned = TabularRewardFunction(
        env_true.env,
        torch.randn([16], device=device, dtype=float_type))

    def trf(env, fun):
        return TabularRewardFunction(env.env, fun(env.parameters))

    metric = RewardL2Loss(env_true)

    for use_sampling in [True, False]:
        kw = dict(gamma=gamma, use_sampling=use_sampling, n_iters=5000)

        def meval(reward, **kwargs):
            for key, value in kw.items():
                if key not in kwargs:
                    kwargs[key] = value
            return metric.evaluate(reward, **kwargs).detach().numpy()

        assert np.allclose(meval(env_true, n_iters=10000), 0, atol=1e-3 if use_sampling else 1e-8), (
            f"Distance from true reward to itself is 0 "
            f"(use_sampling={use_sampling})")
        assert meval(trf(env_true, lambda p: -p)) > 0.1, (
            f"Distance from true reward to itself negative is >0 "
            f"(use_sampling={use_sampling})")

        assert np.allclose(
            meval(env_learned),
            meval(trf(env_learned, lambda p: 5*p), n_iters=10000), rtol=0.01 if use_sampling else 1e-6), (
                "scaling learned reward doesn't change result "
                f"(use_sampling={use_sampling})")

        assert np.allclose(
            meval(env_learned, n_iters=10000),
            meval(trf(env_learned, lambda p: p + 10), n_iters=10000), rtol=1e-2 if use_sampling else 1e-6), (
                "biasing learned reward doesn't change result "
                f"(use_sampling={use_sampling})")

        assert np.allclose(
            meval(env_learned, n_iters=10000),
            meval(trf(env_learned, lambda p: p - 10), n_iters=10000), rtol=1e-2 if use_sampling else 1e-6), (
                "biasing learned reward doesn't change result "
                f"(use_sampling={use_sampling})")


