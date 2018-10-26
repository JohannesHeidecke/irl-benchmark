import numpy as np

# TODO(ao) remove this file and use new


def mellowmax(values, omega=1):
    """
    This paper reports mellowmax has desirable properties that softmax doesn't.

    https://arxiv.org/pdf/1612.05628.pdf
    :param values: 1-D numpy array :TODO: check if we need for higher dimensional
    :param omega: the mellowmax parameter
    """

    mm = (np.log(np.sum(np.exp(omega * values))) - np.log(len(values))) / omega
    return mm


class GoodValueIteration():
    def __init__(self, transition_dynamics, gamma, error=0.01):
        self.transition_dynamics = transition_dynamics
        self.n_states, self.n_actions, _ = np.shape(self.transition_dynamics)
        self.gamma = gamma
        self.error = error

    def train(self, rewards, temperature=None, threshold=1e-10
              ):  # 'train' really doesn't seem like a good naming here.
        """
        Does value iteration on the given rewards, based on the transition dynamics.
        :param rewards:
        :param temperature: Rationality parameter. If 'None', computes completely rationally.
        :param threshold:
        :return: state value function, state-action value function, stochastic policy
        """

        v = np.copy(rewards)

        diff = float("inf")
        if temperature is None:
            # calculate optimal policy
            while diff > threshold:
                v_prev = np.copy(v)

                # calculate q fucntion
                q = rewards.reshape((-1, 1)) + self.gamma * np.dot(
                    self.transition_dynamics, v_prev)

                # v = max(q)
                v = np.amax(q, axis=1)

                diff = np.amax(abs(v_prev - v))

            v = v.reshape((-1, 1))

            # Compute stochastic policy
            # Assigns equal probability to taking actions whose Q_sa == max_a(Q_sa)
            max_q_index = (q == np.tile(
                np.amax(q, axis=1), (self.n_actions, 1)).T)
            policy = max_q_index / np.sum(max_q_index, axis=1).reshape((-1, 1))

        else:

            # calulate Boltzmann-optimal policy using the temperature as the mellowmax parameter

            while diff > threshold:
                v_prev = np.copy(v)

                # calculate q fucntion
                q = rewards.reshape((-1, 1)) + self.gamma * np.dot(
                    self.transition_dynamics, v_prev)

                # v = max(q)
                v = mellowmax(q, temperature)

                diff = np.amax(abs(v_prev - v))

            v = v.reshape((-1, 1))

            # Compute stochastic policy

            expt = lambda x: np.exp(x / temperature)
            tlog = lambda x: temperature * np.log(x)

            # policy_{s,a} = exp((Q_{s,a} - V_s - t*log(nA))/t)
            policy = expt(q - v - tlog(q.shape[1]))

        return v, q, policy
