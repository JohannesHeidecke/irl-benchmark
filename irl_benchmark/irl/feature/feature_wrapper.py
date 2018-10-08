import gym
import numpy as np

from irl_benchmark.utils import to_one_hot

class FeatureWrapper(gym.Wrapper):
    
    def __init__(self, env):
        super(FeatureWrapper, self).__init__(env)
    
    def reset(self):
        self.current_state = self.env.reset()
        return self.current_state
    
    def step(self, action):
        # execute action
        next_state, reward, terminated, info = self.env.step(action)

        info['features'] = self.features(self.current_state, action, next_state)

        # remember which state we are in:
        self.current_state = next_state
        
        return next_state, reward, terminated, info

    def features(self, current_state, action, next_state):
        raise NotImplementedError()

    def feature_shape(self):
        raise NotImplementedError()
    

class FrozenFeatureWrapper(FeatureWrapper):

    def features(self, current_state, action, next_state):
        return to_one_hot(next_state, self.env.observation_space.n)

    def feature_shape(self):
        return (self.env.observation_space.n,)
