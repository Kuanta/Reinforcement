'''
Environment for input filtering. The aim is to design an input filter with the actor network that tries to adjust the reference
of a system by comparing the outputs of a well controlled system.
'''
from Environments.Environment import Environment
import numpy as np

class FilterEnv(Environment):
    def __init__(self, inputs: np.ndarray, outputs: np.ndarray):
        '''
        Constructor
        :param inputs (np.array): One dimensional numpy array for the input values of ref. system
        :param outputs (np.array): One dimensional numpy array for the output values of ref. system
        '''
        self.inputs = inputs
        self.outputs = outputs
        self.preds = None
        self.actions = None
        self.curr_t = 0
        self.reward_weight = 0  # This weight scales the second term in the reward (-(action[t]-action[t-1])^2)

    def reset(self):
        self.preds = np.zeros(len(self.outputs))
        self.actions = np.zeros(len(self.outputs))
        self.preds[0] = self.outputs[0]
        self.actions[0] = 0.5
        self.curr_t = 1 #Start from 1 step ahead
        init_observartion = self.get_observation()
        return init_observartion

    def render(self):
        pass

    def get_observation(self):
        return np.array([self.inputs[self.curr_t-1]])

    def step(self, action):
        done = False

        # Predict
        self.predict(action)

        # Calculate the reward with the current prediction and the current output of the reference system
        reward = -1 * np.power(self.preds[self.curr_t]-self.outputs[self.curr_t], 2)
        reward -= self.reward_weight*np.power(self.actions[self.curr_t]-self.actions[self.curr_t-1], 2)   #TODO: Try this if actions have high variance
        observation = self.get_observation()
        self.curr_t = self.curr_t + 1
        if self.curr_t + 1>= len(self.outputs):
            self.predict(None)
            done = True

        return observation, reward, done

    def predict(self, action):
        if action is not None:
            self.actions[self.curr_t-1] = action  #TODO: Decide whether calculated action should be (t-1) or (t)
        self.preds[self.curr_t] = self.preds[self.curr_t-1]*0.9048 + self.actions[self.curr_t-1]*0.1903



