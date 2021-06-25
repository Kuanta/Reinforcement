import gym
import cv2 as cv
import numpy as np
import torch

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
        resized = cv.resize(obs[0], dsize=(self.width, self.height), interpolation=cv.INTER_CUBIC)
        obs[0] = resized
        return obs

class SwapDimensionsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env

    def observation(self, obs):
        obs[0] = np.transpose(obs[0], (2,0,1))
        return obs

class ImageNormalizeWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env

    def observation(self, obs: np.array):
        obs[0] = obs[0].astype(float) / 255.0
        return obs

class EncoderWrapper(gym.ObservationWrapper):
    def __init__(self, env, encoder, use_gpu=False):
        super().__init__(env)
        self.env = env
        self.encoder = encoder
        # Freeze Encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        if use_gpu:
            self.device = "cuda:0"
        else:
            self.device = "cpu"
        self.encoder.to(self.device)

    def observation(self, obs: list):
        with torch.no_grad():
            obs[0] = self.encoder.encode(obs[0].unsqueeze(0).float()).squeeze(0)
        obs = torch.cat(obs)
        return obs

class TorchifyWrapper(gym.ObservationWrapper):
    def __init__(self, env, use_gpu=False):
        super().__init__(env)
        self.env = env
        if use_gpu:
            self.device = "cuda:0"
        else:
            self.device = "cpu"

    def observation(self, obs: list):
        for i in range(len(obs)):
            obs[i] = torch.tensor(obs[i]).float().to(self.device).squeeze(0)
        return obs

class ActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super(ActionWrapper, self).__init__(env)

    def action(self, action):
        action_ = [1/(1+np.exp(-action[0])), action[1]]
        return action_

class DtRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super(DtRewardWrapper, self).__init__(env)

    def reward(self, reward):
        if reward == -1000:
            reward = -10
        elif reward > 0:
            reward += 10
        else:
            reward += 4

        return reward
