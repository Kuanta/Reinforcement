'''
Environment for input filtering. The aim is to design an input filter with the actor network that tries to adjust the reference
of a system by comparing the outputs of a well controlled system.
'''
from Environments.Environment import Environment
import numpy as np
from enum import Enum

class FilterArchitecture(Enum):
    PARALLEL = 0
    SERIES = 1
    COMBINED = 2


class FilterEnvOptions:
    def __init__(self):
        self.architecture = FilterArchitecture.SERIES


class FilterEnv(Environment):
    def __init__(self, inputs: np.ndarray, outputs: np.ndarray, opts: FilterEnvOptions=FilterEnvOptions()):
        '''
        Constructor
        :param inputs (np.array): One dimensional numpy array for the input values of ref. system
        :param outputs (np.array): One dimensional numpy array for the output values of ref. system
        '''
        self.inputs = inputs
        self.outputs = outputs
        self.opts = opts
        self.preds = None
        self.actions = None
        self.curr_t = 0
        self.act_punish_rate = 0  # This weight scales the second term in the reward (-(action[t]-action[t-1])^2)

        if self.opts.architecture == FilterArchitecture.SERIES:
            self.obs_dim = 3
        elif self.opts.architecture == FilterArchitecture.PARALLEL:
            self.obs_dim = 1

    def reset(self):
        self.preds = np.zeros(len(self.outputs))
        self.actions = np.zeros(len(self.inputs))
        self.curr_t = 1 #Start from 1 step ahead
        init_observartion = self.get_observation()
        return init_observartion

    def render(self):
        pass

    def get_observation(self):
        if self.opts.architecture == FilterArchitecture.PARALLEL:
            return np.array([self.inputs[self.curr_t-1]])
        elif self.opts.architecture == FilterArchitecture.SERIES:
            return np.array([self.outputs[self.curr_t] - self.preds[self.curr_t], self.outputs[self.curr_t-1] - self.preds[self.curr_t-1], self.actions[self.curr_t-1]])
        else:
            pass

    def step(self, action):
        done = False

        # Predict
        self.predict(action)

        # Calculate the reward with the current prediction and the current output of the reference system
        reward = -1 * np.power(self.preds[self.curr_t]-self.outputs[self.curr_t], 2)
        reward -= self.act_punish_rate*np.sqrt(np.power(action, 2))   #TODO: Try this if actions have high variance
        observation = self.get_observation()
        self.curr_t = self.curr_t + 1
        if self.curr_t + 1 >= len(self.outputs):
            self.predict(None) # TODO: Think about this
            done = True

        return observation, reward, done

    def predict(self, action):
        if action is None:
            action = self.actions[self.curr_t-1]
        else:
            if self.opts.architecture == FilterArchitecture.SERIES:
                self.actions[self.curr_t] = action  # TODO: Decide whether calculated action should be (t-1) or (t)
            elif self.opts.architecture == FilterArchitecture.PARALLEL:
                self.actions[self.curr_t] = action

        self.preds[self.curr_t] = self.preds[self.curr_t-1]*0.9048 + action*0.1903



