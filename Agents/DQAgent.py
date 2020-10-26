'''
Deep Q-Learning Agent
'''

from Agents.Agent import Agent
import Agents.ExperienceBuffer as exp
import numpy as np


class DQAgentOpts:
    def __init__(self):
        self.exp_buffer_size = 100
        self.epsilon = 0.1  # Exploration rate for the behaviour policy
        self.epsilon_decay = 0.99  # At each learning step, this value will be multiplied with the epsilon rate


class DQAgent(Agent):
    def __init__(self, network, opts=DQAgentOpts()):
        super().__init__()
        self.network = network
        self.opts = opts
        if self.opts.exp_buffer_size > 0:
            self.exp_buffer = exp.ExperienceBuffer(size=opts.exp_buffer_size)

    def act(self, new_state):
        if np.random.random() > self.opts.epsilon:
            pass

    def learn(self, environment):
        pass
