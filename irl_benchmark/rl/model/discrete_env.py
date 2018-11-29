from gym.envs.toy_text.discrete import DiscreteEnv
import numpy as np

from irl_benchmark.irl.reward.reward_wrapper import RewardWrapper
from irl_benchmark.rl.model.model_wrapper import BaseWorldModelWrapper
from irl_benchmark.utils.wrapper import is_unwrappable_to, unwrap_env


class DiscreteEnvModelWrapper(BaseWorldModelWrapper):
    def __init__(self, env):
        assert is_unwrappable_to(env, DiscreteEnv)
        super(DiscreteEnvModelWrapper, self).__init__(env)

    def n_states(self):
        return self.env.observation_space.n

    def state_to_index(self, state):
        assert np.isscalar(state)
        return state

    def index_to_state(self, index):
        assert np.isscalar(index)
        return index

    def get_transition_array(self):
        env = unwrap_env(self.env, DiscreteEnv)

        # adding +1 to account for absorbing state
        # (reached whenever game ended)
        n_states = env.observation_space.n + 1
        n_actions = env.action_space.n

        transitions = np.zeros([n_states, n_actions, n_states])

        # iterate over all "from" states:
        for state, transitions_given_state in env.P.items():
            # iterate over all actions:
            for action, outcomes in transitions_given_state.items():
                # iterate over all possible outcomes:
                for probability, next_state, _, done in outcomes:
                    # add transition probability T(s, a, s')
                    transitions[state, action, next_state] += probability
                    if done:
                        # outcome was marked as ending the game.
                        # if game is done and state == next_state, map to absorbing state instead
                        if state == next_state:
                            transitions[state, action, next_state] = 0
                        # map next state to absorbing state
                        # make sure that next state wasn't mapped to any other state yet
                        assert np.sum(transitions[next_state, :, :-1]) == 0
                        transitions[next_state, :, -1] = 1.0

        # specify transition probabilities for absorbing state:
        # returning to itself for all actions.
        transitions[-1, :, -1] = 1.0

        return transitions

    def get_reward_array(self):
        env = unwrap_env(self.env, DiscreteEnv)

        # adding +1 to account for absorbing state
        # (reached whenever game ended)
        n_states = env.observation_space.n + 1
        n_actions = env.action_space.n

        if is_unwrappable_to(self.env, RewardWrapper):
            # get the reward function:
            reward_wrapper = unwrap_env(self.env, RewardWrapper)
            reward_function = reward_wrapper.reward_function
        else:
            reward_function = None

        rewards = np.zeros([n_states, n_actions])

        # iterate over all "from" states:
        for state, transitions_given_state in env.P.items():
            # iterate over all actions:
            for action, outcomes in transitions_given_state.items():
                # iterate over all possible outcomes:
                for probability, next_state, reward, done in outcomes:
                    if reward_function is not None:
                        if done and state == next_state:
                            # don't output reward for reaching state if game is over
                            # and already in that state.
                            reward = 0
                        else:
                            rew_input = reward_wrapper.get_reward_input_for(
                                state, action, next_state)
                            reward = reward_function.reward(rew_input)
                    rewards[state, action] += reward * probability

        # reward of absorbing state is zero:
        rewards[-1, :] = 0.0

        return rewards
