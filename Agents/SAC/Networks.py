from Environments.Environment import ContinuousDefinition
from math import gamma
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from collections import namedtuple
import copy
import itertools
from common import polyak_update, freeze_network

# Template multihead network
SACNetworks = namedtuple("SACNetworks", "base critic_1 critic_2 target_critic_1 target_critic_2 policy")

class MultiheadNetwork(nn.Module):
    def __init__(self, networks: SACNetworks, reparam_noise=1e-6):
        super(MultiheadNetwork, self).__init__()
        self.base_net = networks.base
        self.critic_1_net = networks.critic_1
        self.critic_2_net = networks.critic_2
        self.target_critic_1_net = networks.target_critic_1
        self.target_critic_2_net = networks.target_critic_2
        self.policy_net = networks.policy
        self.reparam_noise=reparam_noise

        freeze_network(self.target_critic_1_net)
        freeze_network(self.target_critic_2_net)

    def init_network(self, learning_rate):
        
        if self.base_net is not None:
            self.base_optimizer = torch.optim.Adam(self.base_net.parameters(), learning_rate)
        else:
            self.base_optimizer = None
        critic_params = itertools.chain(self.critic_1_net.parameters(), self.critic_2_net.parameters())
        self.critic_optimizer = torch.optim.Adam(critic_params, learning_rate)
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), learning_rate)
        
        self.update_targets(1)


    def forward(self, x, hidden_states):
        return self.feature_extraction(x, hidden_states)

    def feature_extraction(self, x):
        if self.base_net is not None:
            return self.base_net(x).view(x.shape[0], -1)
        else:
            return x

    def get_critics(self, features, actions):
        critic_1 = self.critic_1_net(features, actions)
        critic_2 = self.critic_2_net(features, actions)
        return critic_1, critic_2
    
    def get_target_critics(self, features, actions):
        critic_1 = self.target_critic_1_net(features, actions)
        critic_2 = self.target_critic_2_net(features, actions)
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
        self.critic_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        self.policy_optimizer.zero_grad()
    
    def update_targets(self, tau):
        polyak_update(self.target_critic_1_net, self.critic_1_net, tau)
        polyak_update(self.target_critic_2_net, self.critic_2_net, tau)
        
class SequentialMultiheadNetwork(MultiheadNetwork):
    def __init__(self, networks: SACNetworks, reparam_noise=1e-6):
        super().__init__(networks, reparam_noise)
        self.hx = None
    
    def feature_extraction(self, x, hx):
        # X must be in the shape of [batch, input_size]
        if self.base_net is not None and hx is None:
            _hx = torch.zeros(1, x.shape[1], self.base_net.hidden_size).to(x.device).float()
            _cx = torch.zeros(1, x.shape[1], self.base_net.hidden_size).to(x.device).float()
            hx = (_hx, _cx)
            x, hx = self.base_net(x, hx)

        return x, hx
    
    def get_critics(self, features, actions, hx):
        critic_1, _ = self.critic_1_net(features, actions, hx)
        critic_2, _ = self.critic_2_net(features, actions, hx)
        return critic_1, critic_2
    
    def get_target_critics(self, features, actions, hx):
        critic_1, _ = self.target_critic_1_net(features, actions, hx)
        critic_2, _ = self.target_critic_2_net(features, actions, hx)
        return critic_1, critic_2

    def sample(self, features, hidden, add_noise=False):
        mu, log_sigma, hidden = self.policy_net(features, hidden)
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
        log_probs = log_probs.sum(2, keepdim=True)
        
        return normalized_actions, log_probs, torch.tanh(mu), hidden
    


