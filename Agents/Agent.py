'''
Abstract Agent Class
'''
import typing
import numpy as np
from Environments.Environment import Environment

Action = typing.NewType("Action", np.ndarray)

class Agent:
    def __init__(self):
        self.current_state = None
    
    def act(self, new_state) -> Action:
        '''
        Call this method to create a new action with a new state
        :param new_state:
        :return: action
        '''
        raise NotImplementedError

    def learn(self, environment: Environment, n_episodes: int):
        '''
        This method will be called from the trainer to update the parameters or tabular values of the agent.
        Loss values should be calculated here.
        Will be called at each epoch by the trainer. All the necessary calculations will be handled by the agent
        :return:
        '''
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def save_model(self, PATH):
        raise NotImplementedError

    def load_model(self, PATH):
        raise NotImplementedError


