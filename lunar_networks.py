from os import stat
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from Environments.Environment import ContinuousDefinition
from Agents.SAC.Networks import MultiheadNetwork, SequentialMultiheadNetwork, SACNetworks

def init_weight(m):
    if type(m) == nn.Linear:
        m.weight.data.uniform_(-3e-3,3e-3)
        m.bias.data.uniform_(-3e-3,3e-3)

class BaseNet(nn.Module):
    def __init__(self, input_size, hidden_size=128, layer_size=1):
        super(BaseNet, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size)
        self.hx = None
    def forward(self, x, hiddens):

        out, (hx, cx) = self.lstm(x, hiddens)
        return out[-1], (hx, cx)

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
    def __init__(self, state_size, action_size, hidden_size=256):
        super(CriticNet, self).__init__()

        self.fc1 = nn.Linear(state_size+action_size, 256)
        self.lstm = nn.LSTM(input_size=256, hidden_size=hidden_size)
        self.fc2 = nn.Linear(256, 1)
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size

    def forward(self, states, actions, hidden):

        x = torch.cat([states, actions], 2)
        x = self.fc1(x)
        x = F.relu(x)
        x, hidden = self.lstm(x, hidden)
        x = F.relu(x)
        x = self.fc2(x)
        return x, hidden

class PolicyNet(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.lstm = nn.LSTM(input_size=256, hidden_size=hidden_size)
        self.fc2 = nn.Linear(256, 256)
        self.mu_layer = nn.Linear(256, action_size)
        self.sigma_layer = nn.Linear(256, action_size)
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
    
    def forward(self, x, hidden):
        if hidden is None:
            _hx = torch.zeros(1, x.shape[1], self.hidden_size).float().to(x.device)
            _cx = torch.zeros(1, x.shape[1], self.hidden_size).float().to(x.device)
            hidden = (_hx, _cx)
        x = self.fc1(x)
        x = F.relu(x)
        x, hidden = self.lstm(x, hidden)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        mu = self.mu_layer(x)
        log_sigma = self.sigma_layer(x)
        log_sigma = torch.clamp(log_sigma, -20, 2)
        return mu, log_sigma, hidden

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

class LunarNetworkLSTM(SequentialMultiheadNetwork):

    def __init__(self, state_size, hidden_size, action_size, init_w = 3e-3):
        self.reparam_noise = 1e-6
        critic_1_net = CriticNet(state_size, action_size)
        critic_2_net = CriticNet(state_size, action_size)
        target_critic_1_net = CriticNet(state_size, action_size)
        target_critic_2_net = CriticNet(state_size, action_size)
        policy_net = PolicyNet(state_size, action_size)

        networks = SACNetworks(None, critic_1_net, critic_2_net, target_critic_1_net, target_critic_2_net, policy_net)
        super().__init__(networks)