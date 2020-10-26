'''
REINFORCE Agent
'''

from Agents.Agent import Agent
import Agents.ExperienceBuffer as exp
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from common import calc_returns

class ReinforceAgentOpts:
    def __init__(self):
        self.learning_rate = 0.001
        self.optimizer = torch.optim.Adam

class ReinforceAgent(Agent):
    def __init__(self, network: nn.Module, opts=ReinforceAgentOpts()):
        super().__init__()
        self.network = network
        self.opts = opts

        # Parse Opts
        self.optimizer = self.opts.optimizer(self.network.parameters(), lr=self.opts.learning_rate)

    def act(self, new_state):
        new_state = torch.from_numpy(new_state).float().unsqueeze(0)
        probs = self.network.forward(new_state)
        m = Categorical(probs)
        action = m.sample()
        log_prob = m.log_prob(action)
        return action.item(), log_prob

    def learn(self, environment, n_episodes):
        total_loss = []
        for _ in range(n_episodes):
            curr_state = environment.reset()
            states = []
            actions = []
            rewards = []
            log_probs = []
            total_reward = 0
            while True:
                states.append(curr_state)
                action, log_prob = self.act(curr_state)
                log_probs.append(log_prob)
                actions.append(action)
                obs, reward, done, _ = environment.step(action)
                total_reward += reward
                rewards.append(reward)
                curr_state = obs
                if done:
                    break
            print("Episode return:%f" % (total_reward))
            returns = torch.tensor(calc_returns(rewards))
            returns = (returns - returns.mean()) / (returns.std() + 0.00001)
            episode_loss = torch.sum(torch.cat(log_probs) * returns)
            total_loss.append((-1 * episode_loss).float())

        self.optimizer.zero_grad()
        loss = torch.sum(torch.stack(total_loss)).float()
        loss.backward()
        self.optimizer.step()

    def save_model(self, path: str):
        torch.save(self.network.state_dict(), path)

    def load_model(self, path: str):
        self.network.load_state_dict(torch.load(path))
