from Environments.Environment import ContinuousDefinition
from math import gamma
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from collections import namedtuple
import copy
from common import polyak_update

# Template multihead network
SACNetworks = namedtuple("SACNetworks", "base value target_value critic_1 critic_2 policy")

class MultiheadNetwork(nn.Module):
    def __init__(self, networks: SACNetworks):
        super(MultiheadNetwork, self).__init__()
        self.base_net = networks.base
        self.value_net = networks.value
        self.target_value_net = networks.target_value
        self.critic_1_net = networks.critic_1
        self.critic_2_net = networks.critic_2
        self.policy_net = networks.policy

    def init_network(self, learning_rate):
        
        if self.base_net is not None:
            self.base_optimizer = torch.optim.Adam(self.base_net.parameters(), learning_rate)
        else:
            self.base_optimizer = None
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), learning_rate)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1_net.parameters(), learning_rate)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2_net.parameters(), learning_rate)
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), learning_rate)
        
        polyak_update(self.target_value_net, self.value_net, 1)


    def forward(self, x):
        return self.feature_extraction(x)

    def feature_extraction(self, x):
        if self.base_net is not None:
            return self.base_net(x)
        else:
            return x
    
    def get_value(self, features):
        return self.value_net(features)

    def get_target_value(self, features):
        with torch.no_grad():
            return self.target_value_net(features)

    def get_critics(self, features, actions):
        x = torch.cat([features,actions],1)
        critic_1 = self.critic_1_net(x)
        critic_2 = self.critic_2_net(x)
        return critic_1, critic_2
    
    def sample(self, features, add_noise=False):
        mu, log_sigma = self.policy_net(features)
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

    def zero_grads(self):
        if self.base_optimizer is not None:
            self.base_optimizer.zero_grad()
        self.value_optimizer.zero_grad()
        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        self.policy_optimizer.zero_grad()
        

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


