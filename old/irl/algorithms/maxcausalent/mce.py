import gym
import numpy as np
from gym.envs.toy_text.discrete import DiscreteEnv

from irl_benchmark.irl.algorithms.base_algorithm import BaseIRLAlgorithm
from irl_benchmark.rl.algorithms.val_iter import GoodValueIteration
from irl_benchmark.utils.utils import unwrap_env, is_unwrappable_to


class MaxCausalEnt(BaseIRLAlgorithm):
    '''Maximum Entropy IRL (Ziebart et al., 2008).

    Not to be confused with Maximum Entropy Deep IRL (Wulfmeier et al., 2016)
    or Maximum Causal Entropy IRL (Ziebart et al., 2010).
    '''

    def __init__(self,
                 env,
                 expert_trajs,
                 transition_dynamics,
                 feat_mat,
                 rl_alg_factory,
                 temperature=1,
                 gamma=0.99,
                 lr=0.2,
                 epochs=1
                 ):
        '''Initialize Maximum Entropy IRL algorithm. '''
        super(MaxCausalEnt, self).__init__(env, expert_trajs, rl_alg_factory)
        # make sure env is DiscreteEnv (other cases not implemented yet
        # TODO: implement other cases
        assert is_unwrappable_to(env, DiscreteEnv)
        self.base_env = unwrap_env(env, gym.envs.toy_text.discrete.DiscreteEnv)

        self.gamma = gamma
        self.transition_dynamics = transition_dynamics
        self.n_states, self.n_actions, _ = np.shape(self.transition_dynamics)
        self.lr = lr
        self.feat_mat = feat_mat
        self.temperature = temperature
        self.epochs = epochs

    def sa_visitations(self):
        """
        Given a list of trajectories in an mdp, computes the state-action
        visitation counts and the probability of a trajectory starting in state s.

        ----------
        env: object
            Instance of the MDP class.
        gamma : float
            Discount factor; 0<=gamma<=1.
        trajectories : 3D numpy array
            Expert trajectories.
        Returns
        -------
        (2D numpy array, 1D numpy array)
            Arrays of shape (n_states, n_actions) and (n_states).
        """

        s0_count = np.zeros(self.n_states)
        sa_visit_count = np.zeros((self.n_states, self.n_actions))

        for traj in self.expert_trajs:
            # traj['states'][0] is the state of the first timestep of the trajectory.
            s0_count[traj['states'][0]] += 1

            for timestep in len(traj['states']):
                state = traj['states'][timestep]
                action = traj['actions'][timestep]

                sa_visit_count[state, action] += 1

        # Count into probability
        P0 = s0_count / len(self.expert_trajs)

        return sa_visit_count, P0

    def occupancy_measure(self, policy, P0, t_max=None, threshold=1e-6):

        """
        Computes occupancy measure of a MDP under a given time-constrained policy
        -- the expected discounted number of times that policy π visits state s in
        a given number of timesteps, as in Algorithm 9.3 of Ziebart's thesis:
        http://www.cs.cmu.edu/~bziebart/publications/thesis-bziebart.pdf.

        ----------

        policy : 2D numpy array
            policy[s,a] is the probability of taking action a in state s.
        P0 : 1D numpy array of shape (n_states)
            i-th element is the probability that the traj will start in state i.
        t_max : int
            number of timesteps the policy is executed.
        Returns
        -------
        1D numpy array of shape (mdp.nS)
        """

        if P0 is None:
            P0 = np.ones(self.n_states) / self.n_states
        d_prev = np.zeros_like(P0)

        t = 0

        diff = float("inf")

        while diff > threshold:

            d = np.copy(P0)

            for state in range(self.n_states):
                for action in range(self.n_actions):
                    # for all next_state reachable:

                    for next_state in range(self.n_states):
                        # probabilty of reaching next_state by taking action
                        prob = self.transition_dynamics[state, action, next_state]
                        d[next_state] += self.gamma * d_prev[state] * policy[state, action] * prob

            diff = np.amax(abs(d_prev - d))  # maxima of the flattened array
            d_prev = np.copy(d)

            if t_max is not None:
                t += 1
                if t == t_max:
                    break
        return d

    def train(self, time_limit=300, rl_time_per_iteration=30):

        """
        Finds theta, a reward parametrization vector (r[s] = features[s]'.*theta)
        that maximizes the log likelihood of the given expert trajectories,
        modelling the expert as a Boltzmann rational agent with given temperature.

        This is equivalent to finding a reward parametrization vector giving rise
        to a reward vector giving rise to Boltzmann rational policy whose expected
        feature count matches the average feature count of the given expert
        trajectories (Levine et al, supplement to the GPIRL paper).
        Parameters

        We finally return rewards (non-normalized).
        """

        sa_visit_count, P0 = self.sa_visitations()

        mean_s_visit_count = np.sum(sa_visit_count, 1) / len(self.expert_trajs)

        mean_feature_count = np.dot(self.feat_mat.T, mean_s_visit_count)

        # initialize the parameters
        theta = np.random.rand(self.feat_mat.shape[1])
        solver = GoodValueIteration(transition_dynamics=self.transition_dynamics, gamma=self.gamma)
        for i in range(self.epochs):
            r = np.dot(self.feat_mat, theta)

            # compute the policy :TODO: implement Boltzmann version of VI
            v, q, policy = solver.train(rewards=r, temperature=None)

            # Log-Likelihood
            l = np.sum(sa_visit_count * (q - v))  # check: broadcasting works as intended or not

            # occupancy measure
            d = self.occupancy_measure(policy=policy, P0=P0)

            # log-likeilihood gradient
            dl_dtheta = -(mean_feature_count - np.dot(self.feat_mat.T, d))

            # graduate descent
            theta -= self.lr * dl_dtheta

            # report stuff every
            if (i + 1) % 10 == 0:
                print('Epoch: {} log likelihood of all traj: {}'.format(i, l))

        estimated_rewards = np.dot(self.feat_mat, theta)
        print(estimated_rewards)
        return estimated_rewards
