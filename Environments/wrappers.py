import gym
import cv2 as cv
import numpy as np

class ResizeWrapper(gym.ObservationWrapper):
    '''
    Resizes an observation to the desired dimenisons. Observations must be in the for of (height, width, channels)
    '''
    def __init__(self, env, width, height):
        super().__init__(env)
        self.env = env
        self.width = width
        self.height = height
    
    def observation(self, obs: np.array):
        '''
        In the pytorch setting, channels must be the first dimension, so we need to swap the axes
        '''
        resized = cv.resize(obs, dsize=(self.width, self.height), interpolation=cv.INTER_CUBIC)
        return resized

class SwapDimensionsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env

    def observation(self, obs):
        return np.transpose(obs, (2,0,1))

class ImageNormalizeWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env

    def observation(self, obs: np.array):
        return obs.astype(float) / 255.0