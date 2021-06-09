'''
Defines a Trainer class. Trainer class trains an agent using an environment
'''

import json
import torch.optim as optim
from Agents.Agent import Agent
from Environments.Environment import Environment
import os


class TrainOpts:
    def __init__(self):
        self.n_epochs = 100
        self.optimizer = optim.Adam
        self.episodic = True
        self.n_episodes = 100  # Number of episodes to run before calling the learn method of the agent
        self.n_iterations = 100
        self.save_path = "./tmp"


class Trainer:
    def __init__(self, agent: Agent, env: Environment, opts=TrainOpts()):
        self.agent = agent
        self.env = env
        self.opts = opts

    def train(self):
        if self.opts.episodic:
            all_rewards, avg_rewards = self.agent.learn(self.env, self.opts.n_episodes, self.opts.n_iterations)
            info = {"Rewards":all_rewards, "Averages":avg_rewards}
            if not os.path.exists(self.opts.save_path):
                os.mkdir(self.opts.save_path)
            with open(os.path.join(self.opts.save_path, "rewards"), 'w') as fp:
                json.dump(info, fp)
        else:
            # TODO: Implement continuous training
            pass

    def test(self):
        if self.opts.episodic:
            curr_state = self.env.reset()
            while True:
                action = self.agent.act(curr_state, evaluation=True)
                obs, rew, done, _ = self.env.step(action.cpu().detach().numpy())
                curr_state = obs
                self.env.render()
                if done:
                    break

