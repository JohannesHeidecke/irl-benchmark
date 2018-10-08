import gym

from irl_benchmark.irl_benchmark.irl.reward.reward_function import State, StateAction, StateActionState, FeatureBasedRewardFunction


class RewardWrapper(gym.Wrapper):

    def __init__(self, env, reward_function):
        super(RewardWrapper, self).__init__(env)
        self.reward_function = reward_function

    def reset(self):
        self.current_state = self.env.reset()
        return self.current_state

    def step(self, action):
        # execute action
        next_state, reward, terminated, info = self.env.step(action)

        # persist true reward in information:
        info['true_reward'] = reward

        # generate input for reward function:
        if isinstance(self.reward_function, FeatureBasedRewardFunction):
            rew_input = info['features']
        else:
            if self.reward_function.action_in_domain:
                if self.reward_function.next_state_in_domain:
                    rew_input = StateActionState(
                        self.current_state, action, next_state)
                else:
                    rew_input = StateAction(self.current_state, action)
            else:
                rew_input = State(self.current_state)

        reward = self.reward_function.reward(rew_input).item()

        # remember which state we are in:
        self.current_state = next_state

        return next_state, reward, terminated, info

    def update_reward_function(self, reward_function):
        self.reward_function = reward_function
