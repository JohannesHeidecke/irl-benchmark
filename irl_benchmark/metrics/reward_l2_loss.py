import numpy as np
import slm_lab.agent.net
import torch
from gym.spaces.discrete import Discrete as DiscreteSpace
from irl_benchmark.utils import utils
from irl_benchmark.irl.reward.reward_function import (AbstractRewardFunction,
    State, StateActionState, TabularRewardFunction)
from irl_benchmark.utils.utils import unwrap_env
from slm_lab.lib import logger


__all__ = ['RewardL2Loss']

logger = logger.get_logger(__name__)

_env_net_specs = {
    "FrozenLake-v0": {
        "type": "MLPNet",
        "shared": False,
        "hid_layers": [64],
        "hid_layers_activation": "selu",
        "clip_grad": False,
        "use_same_optim": True,
        "optim_spec": {
          "name": "Adam",
          "lr": 0.005
        },
        "lr_decay": "rate_decay",
        "lr_decay_frequency": 200,
        "lr_decay_min_timestep": 200,
        "lr_anneal_timestep": 10000,
        "batch_size": 64,
    },
    "FrozenLake8x8-v0": {
        "type": "MLPNet",
        "shared": False,
        "hid_layers": [64],
        "hid_layers_activation": "selu",
        "clip_grad": False,
        "use_same_optim": True,
        "optim_spec": {
          "name": "Adam",
          "lr": 0.005
        },
        "lr_decay": "rate_decay",
        "lr_decay_frequency": 200,
        "lr_decay_min_timestep": 200,
        "lr_anneal_timestep": 10000,
        "batch_size": 64,
    }
}


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
                 gamma: float, use_sampling=False, use_gpu=False, n_iters=5000) -> float:
        '''Return distance between estim_reward and true reward.'''
        if use_gpu and torch.cuda.device_count() > 0 and torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            use_gpu = False
            device = torch.device('cpu')
        # HACK: whether to use GPU or not should go in the Spec.

        if use_sampling:
            env = self.true_reward.env
            net_spec = _env_net_specs[env.spec.id]
            return _optimize_net(env, net_spec, gamma, self.true_reward,
                                 estim_reward, n_iters=n_iters)
        else:
            domain = estim_reward.domain()
            if not isinstance(domain, State):
                raise NotImplementedError(
                    "StateAction or StateActionState rewards")
            R_true = self.true_reward.reward(domain)
            R_est = estim_reward.reward(domain)

            R_true = torch.from_numpy(R_true).to(device=device)
            R_est = torch.from_numpy(R_est).to(device=device)

            phi_true = _opt_potential(R_true, gamma)
            vija_true = _scalable_term(R_true, gamma, phi_true)
            phi_est = _opt_potential(R_est, gamma)
            vija_est = _scalable_term(R_est, gamma, phi_est)

            c_ = (torch.dot(vija_est, vija_true)
                  / torch.dot(vija_est, vija_est))
            # Only nonnegative scalings c are valid
            self._last_c = c = torch.max(
                    input=c_, other=torch.zeros(torch.Size(), dtype=c_.dtype,
                        device=c_.device))
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
    diag_M, off_diag_M = (torch.from_numpy(a).to(R.device)
            for a in _elements_of_M(gamma, N))

    v = (-Ri_ + gamma*R_j)/A
    r = -off_diag_M*v.sum() + (off_diag_M - diag_M)*v
    return r


def _scalable_term(R, gamma, phi):
    R_full = R.reshape((R.shape + (1, 1))[:3])
    return ((R_full - phi.reshape([-1, 1, 1])
             + gamma*phi.reshape([1, 1, -1]))
            .reshape([-1]))


class RewardLossModule(torch.nn.Module):
    def __init__(self, estim_reward, gamma, in_dim, net, net_dtype):
        super(RewardLossModule, self).__init__()
        self.estim_reward = estim_reward
        self.gamma = gamma
        self.in_dim = in_dim
        self.net_dtype = net_dtype

        # child Modules
        self.c = torch.nn.Parameter(torch.ones((), dtype=net_dtype),
                                    requires_grad=True)
        self.net = net

    def reset_parameters(self):
        self.c[...] = 1.0

    @staticmethod
    def correctly_shaped(R, dtype, device):
        return (torch.from_numpy(R)
                .to(dtype=dtype, device=device)
                .reshape((R.shape + (1, 1))[:3]))

    def forward(self, input):
        dev_dtype = dict(device=self.c.device, dtype=self.net_dtype)
        oh_states = utils.to_one_hot(
            input.state, self.in_dim,
            lambda s: torch.zeros(s, **dev_dtype))

        R = self.correctly_shaped(self.estim_reward.reward(input), **dev_dtype)

        phi_s = self.net(oh_states)
        return (self.c**2 * R - phi_s.reshape([-1, 1, 1])
                + self.gamma*phi_s.reshape([1, 1, -1]))


def _optimize_net(env, net_spec, gamma, true_reward, estim_reward, n_iters):
    env = unwrap_env(env)
    if isinstance(env.observation_space, DiscreteSpace):
        in_dim = env.observation_space.n
    else:
        assert len(env.observation_space.shape) == 1, (
            "Cannot handle tensor spaces")
        in_dim = env.observation_space.shape[0]

    NetClass = getattr(slm_lab.agent.net, net_spec['type'])
    net_spec["loss_spec"] = {"name": "MSELoss", "reduction": "sum"}
    net = NetClass(net_spec, in_dim, out_dim=1)
    net_dtype = next(next(net.modules()).parameters()).dtype

    net.model = RewardLossModule(estim_reward, gamma, in_dim, net.model,
                                 net_dtype)
    # Reconstruct optimizer with new set of parameters
    net.optim = slm_lab.agent.net.net_util.get_optim(
        net, net_spec["optim_spec"])

    print("Resulting net:", net)

    for i in range(n_iters):
        # d = true_reward.domain_sample(net_spec['batch_size'])
        sas = true_reward.domain()
        R_true = (RewardLossModule.correctly_shaped(
            true_reward.reward(sas),
            dtype=net_dtype,
            device=net.device)
                  .expand([-1, -1, in_dim]))  # HACK
        loss = net.training_step(x=sas, y=R_true)
        with torch.no_grad():
            torch.clamp(net.model.c, 0., np.inf, out=net.model.c)
    return loss
