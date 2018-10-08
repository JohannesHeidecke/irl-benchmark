import gym
import numpy as np

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
    

class FrozenFeatureWrapper(FeatureWrapper):

    def features(self, current_state, action, next_state):
        features = np.zeros(16)
        features[next_state] = 1.0
        return features
