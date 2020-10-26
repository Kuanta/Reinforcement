'''
Defines a Trainer class. Trainer class trains an agent using an environment
'''

import numpy as np
import torch
import torch.optim as optim
from Agents.Agent import Agent
from Environments.Environment import Environment


class TrainOpts:
    def __init__(self):
        self.n_epochs = 100
        self.optimizer = optim.Adam
        self.episodic = True
        self.n_episodes = 100  # Number of episodes to run before calling the learn method of the agent


class Trainer:
    def __init__(self, agent: Agent, env: Environment, opts=TrainOpts()):
        self.agent = agent
        self.env = env
        self.opts = opts

    def train(self):
        if self.opts.episodic:
            for _ in range(self.opts.n_epochs):
                reward = self.agent.learn(self.env, self.opts.n_episodes)
        else:
            # TODO: Implement continuous training
            pass

    def test(self):
        if self.opts.episodic:
            for _ in range(self.opts.n_episodes):
                curr_state = self.env.reset()
                while True:
                    action, _ = self.agent.act(curr_state)
                    obs, rew, done, _ = self.env.step(action)
                    curr_state = obs
                    self.env.render()
                    if done:
                        break

