from Environments.Environment import ContinuousDefinition
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
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
        self.fc3.weight.data.uniform_(-(3e-3), 3e-3)
        self.fc3.bias.data.uniform_(-(3e-3), 3e-3)
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x


class CriticNet(nn.Module):
    def __init__(self, n_states, n_actions):
        super(CriticNet ,self).__init__()
        self.fc1 = nn.Linear(n_states + n_actions, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, states, actions):
        x = torch.cat([states,actions],1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

class ActorNet(nn.Module):
    def __init__(self, n_states, act_def: ContinuousDefinition, reparam_noise):
        super(ActorNet, self).__init__()
        self.act_def = act_def
        self.fc1 = nn.Linear(n_states, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mean_layer = nn.Linear(256, act_def.shape[0])
        self.sigma_layer = nn.Linear(256, act_def.shape[0])
        self.reparam_noise = reparam_noise

    def forward(self, states):
        x = self.fc1(states)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        
        mu = self.mean_layer(x)
        log_sigma = self.sigma_layer(x)
        log_sigma = torch.clamp(log_sigma, -20, 2)

        return mu, log_sigma

    def sample(self, states, add_noise=False):
        '''
        From the Appendix C of SAC paper:
        log(pi) = log(Gaussian) - sum(log(1-tanh(ui)^2))
        '''
        mu, log_sigma = self.forward(states)
        sigma = log_sigma.exp()
        dist = Normal(mu, sigma)

        if add_noise:
            actions = dist.rsample()
        else:
            actions = dist.sample()

        normalized_actions = torch.tanh(actions)

        # Calculate log probs
        log_probs = dist.log_prob(actions) 
        log_probs -= torch.log(1-normalized_actions.pow(2)+self.reparam_noise)
        log_probs = log_probs.sum(1, keepdim=True)
        
        return normalized_actions, log_probs, torch.tanh(mu)


