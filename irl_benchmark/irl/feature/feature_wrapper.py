import gym
from gym.envs.classic_control.pendulum import PendulumEnv, angle_normalize
import numpy as np

from irl_benchmark.utils.utils import to_one_hot, unwrap_env


class FeatureWrapper(gym.Wrapper):
    '''Provide features in info dictionary.'''

    def __init__(self, env):
        super(FeatureWrapper, self).__init__(env)

    def reset(self):
        '''Reset environment and return initial state.
        No changes to base class reset function.
        '''
        self.current_state = self.env.reset()
        return self.current_state

    def step(self, action):
        '''Call base class step method but also log features.
        Args:
          action: `int` corresponding to action to take
        Returns:
          next_state: from env.observation_space, via base class
          reward: `float`, via base class
          terminated: `bool`, via base class
          info: `dictionary` w/ additional key 'features' compared to
            base class, provided by this class's features method
        '''
        # execute action
        next_state, reward, terminated, info = self.env.step(action)

        info['features'] = self.features(self.current_state, action,
                                         next_state)

        # remember which state we are in:
        self.current_state = next_state

        return next_state, reward, terminated, info

    def features(self, current_state, action, next_state):
        '''Return features to be saved in step method's info dictionary.'''
        raise NotImplementedError()

    def feature_shape(self):
        '''Get shape of features.'''
        raise NotImplementedError()


class FrozenLakeFeatureWrapper(FeatureWrapper):
    '''Feature wrapper that was ad hoc written for the FrozenLake env.
    Would also work to get one-hot features for any other discrete env
    such that feature-based algorithms can be used in a tabular setting.
    '''

    def features(self, current_state, action, next_state):
        '''Return one-hot encoding of next_state.'''
        return to_one_hot(next_state, self.env.observation_space.n)

    def feature_shape(self):
        '''Return dimension of the one-hot vectors used as features.'''
        return (self.env.observation_space.n, )


class PendulumFeatureWrapper(FeatureWrapper):
    def features(self, current_state, action, next_state):
        th, thdot = unwrap_env(self.env.env, PendulumEnv).state
        action = np.clip(action, -self.env.env.max_torque, self.env.env.max_torque)[0]
        return np.array([angle_normalize(th)**2, thdot**2, action**2])

    def feature_shape(self):
        '''Return one-hot encoding of next_state.'''
        return (3, )
