import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from Environments.Environment import ContinuousDefinition
from Agents.SAC.Networks import MultiheadNetwork, SACNetworks

def init_weight(m):
    if type(m) == nn.Linear:
        m.weight.data.uniform_(-3e-3,3e-3)
        m.bias.data.uniform_(-3e-3,3e-3)

class BaseNet(nn.Module):
    def __init__(self, input_size, hidden_size=128, layer_size=1):
        self.lstm = nn.LSTMCell(input_size=input_size, hidden_size=hidden_size)

class ValueNet(nn.Module):
    def __init__(self, state_size):
        super(ValueNet, self).__init__()
        final_layer = nn.Linear(256, 1)
        final_layer.weight.data.uniform_(-(3e-3), 3e-3)
        self.layers = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            final_layer
        )

    def forward(self, x):
        return self.layers(x)

class CriticNet(nn.Module):
    def __init__(self, state_size, action_size):
        super(CriticNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_size+action_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    def forward(self, x):
        return self.layers(x)

class PolicyNet(nn.Module):
    def __init__(self, state_size, action_size):
        super(PolicyNet, self).__init__()
        self.base = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        self.mu_layer = nn.Linear(256, action_size)
        self.sigma_layer = nn.Linear(256, action_size)
    
    def forward(self, x):
        x = self.base(x)
        mu = self.mu_layer(x)
        log_sigma = self.sigma_layer(x)
        log_sigma = torch.clamp(log_sigma, -20, 2)
        return mu, log_sigma

class LunarNetwork(MultiheadNetwork):

    def __init__(self, state_size, action_size, init_w = 3e-3):
        self.reparam_noise = 1e-6
        
        critic_1_net = CriticNet(state_size, action_size)
        critic_2_net = CriticNet(state_size, action_size)
        target_critic_1_net = CriticNet(state_size, action_size)
        target_critic_2_net = CriticNet(state_size, action_size)
        policy_net = PolicyNet(state_size, action_size)

        networks = SACNetworks(None, critic_1_net, critic_2_net, target_critic_1_net, target_critic_2_net, policy_net)
        super().__init__(networks)