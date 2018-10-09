

import numpy as np


from irl_benchmark.rl.algorithms import RLAlgorithm


class ValueIteration(RLAlgorithm):

    def __init__(
        self,
        env,
        transition_dynamics,
        gamma=0.95,
    ):
        self.env = env
        self.gamma = gamma
        self.P = transition_dynamics
        self.n_actions = env.action_space.n

    def value_iteration(self, rewards, error=0.01, deterministic=False):

        n_states, n_actions, _ = np.shape(self.P)

        values = np.zeros([n_states])

        # estimate values
        while True:
            values_tmp = values.copy()

            for s in range(n_states):
                values[s] = max(
                    [sum([self.P[s, a, s1] * (rewards[s] + self.gamma * values_tmp[s1]) for s1 in range(n_states)]) for a in
                     range(n_actions)])

            if max([abs(values[s] - values_tmp[s]) for s in range(n_states)]) < error:
                break

        if deterministic:
            # generate deterministic policy
            policy = np.zeros([n_states])
            for s in range(n_states):
                policy[s] = np.argmax([sum([self.P[s, a, s1] * (rewards[s] + self.gamma * values[s1])
                                            for s1 in range(n_states)])
                                       for a in range(n_actions)])

            return values, policy
        else:
            # generate stochastic policy
            policy = np.zeros([n_states, n_actions])
            for s in range(n_states):
                v_s = np.array([sum([self.P[s, a, s1] * (rewards[s] + self.gamma * values[s1]) for s1 in range(n_states)]) for a in
                                range(n_actions)])
                policy[s, :] = np.transpose(v_s / np.sum(v_s))
        return values, policy
