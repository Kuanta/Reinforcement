'''
Actor Critic Agent.
'''

from Agents.Agent import Agent
import torch
import torch.nn as nn
from torch.distributions import Categorical
from common import calc_returns


class ActorCriticAgentOpts:
    def __init__(self):
        self.learning_rate = 0.001
        self.optimizer = torch.optim.Adam


class ActorCriticAgent():
    def __init__(self, network: nn.Module, opts=ActorCriticAgentOpts()):
        super().__init__()
        self.network = network
        self.opts = opts

        # Parse Opts
        self.optimizer = self.opts.optimizer(self.network.parameters(), lr=self.opts.learning_rate)

    def act(self, new_state):
        new_state = torch.from_numpy(new_state).float().unsqueeze(0)
        actor, critic = self.network.forward(new_state)
        m = Categorical(actor)
        action = m.sample()
        log_prob = m.log_prob(action)
        return action.item(), (log_prob, critic)

    def learn(self, environment, n_episodes):
        total_loss = []
        for _ in range(n_episodes):
            curr_state = environment.reset()
            states = []
            actions = []
            rewards = []
            log_probs = []
            critics = []
            total_reward = 0
            while True:
                states.append(curr_state)
                action, outs = self.act(curr_state)
                log_prob = outs[0]
                critic = outs[1]
                log_probs.append(log_prob)
                actions.append(action)
                critics.append(critic)
                obs, reward, done, _ = environment.step(action)
                total_reward += reward
                rewards.append(reward)
                curr_state = obs
                if done:
                    break
            print("Episode return:%f" % (total_reward))
            critics = torch.cat(critics)
            returns = torch.tensor(calc_returns(rewards))
            advantages = returns - critics
            policy_loss = (-1*torch.cat(log_probs))*(advantages.detach())
            critic_loss = (returns-critics)*(returns-critics)
            episode_loss = policy_loss.sum() + critic_loss.sum()
            total_loss.append(episode_loss.float())

        self.optimizer.zero_grad()
        loss = torch.sum(torch.stack(total_loss)).float()
        loss.backward()
        self.optimizer.step()

    def save_model(self, path: str):
        torch.save(self.network.state_dict(), path)

    def load_model(self, path: str):
        self.network.load_state_dict(torch.load(path))