from math import gamma
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

# Regular Networks

class ValueNet(nn.Module):
    def __init__(self, n_inputs):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(n_inputs, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = F.self.fc3(x)
        return x


class CriticNet(nn.Module):
    def __init__(self, n_states, n_actions):
        super(CriticNet ,self).__init__()
        self.fc1 = nn.Linear(n_states + n_actions, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self,states, actions):
        x = torch.stack(states, actions)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

class ActorNet(nn.Module):
    def __init__(self, n_states, n_actions, max_action, reparam_noise):
        super(ActorNet, self).__init__()
        self.max_action = max_action
        self.fc1 = nn.Linear(n_states, 256)
        self.fc2 = nn.Linear(256, 128)
        self.mean_layer = nn.Linear(128, n_actions)
        self.sigma_layer = nn.Linear(128, n_actions)
        self.reparam_noise = reparam_noise

    def forward(self, states):
        x = self.fc1(states)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        
        mu = self.mean_layer(x)
        sigma = self.sigma_layer(x)
        sigma = torch.clamp(sigma, self.reparam_noise, 1)

        return mu, sigma

    def sample(self, states, add_noise=False):
        '''
        From the Appendix C of SAC paper:
        log(pi) = log(Gaussian) - sum(log(1-tanh(ui)^2))
        '''
        mu, sigma = self.forward(states)
        dist = Normal(mu, sigma)

        if add_noise:
            actions = dist.rsample()
        else:
            actions = dist.sample()

        action = torch.tanh(actions)
        log_probs = dist.log_prob(actions)
        log_probs -= torch.sum(torch.log(1-action.pow(2))+self.reparam_noise)
        
        return action, log_probs 

