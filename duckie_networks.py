from lunar_networks import PolicyNet
from Agents.SACv2.Networks import MultiheadNetwork, SACNetworks
import torch
import torch.nn as nn
import torch.nn.functional as F
 
class BaseNet(nn.Module):
    def __init__(self, in_channels):
        super(BaseNet, self).__init__()
        self.layers = nn.Sequential(nn.Conv2d(in_channels, out_channels=32, kernel_size=4, stride=2),
        nn.ReLU(),
        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2),
        nn.ReLU(),
        nn.Conv2d(in_channels=32, out_channels=16, kernel_size=4, stride=2),
        nn.ReLU())
    
    def forward(self, x):
        out = self.layers(x)
        return out

class ValueNet(nn.Module):
    def __init__(self, n_inputs):
        super(ValueNet, self).__init__()
        self.n_inputs = n_inputs
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
        self.n_states = n_states
        self.fc1 = nn.Linear(n_states + n_actions, 256)
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

class ActorNet(nn.Module):
    def __init__(self, n_states, action_size):
        super(ActorNet, self).__init__()
        self.fc1 = nn.Linear(n_states, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mean_layer = nn.Linear(256, action_size)
        self.sigma_layer = nn.Linear(256, action_size)
        self.mean_layer.weight.data.uniform_(-(3e-3), 3e-3)
        self.mean_layer.bias.data.uniform_(-(3e-3), 3e-3)
        self.sigma_layer.weight.data.uniform_(-(3e-3), 3e-3)
        self.sigma_layer.bias.data.uniform_(-(3e-3), 3e-3)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        
        mu = self.mean_layer(x)
        log_sigma = self.sigma_layer(x)
        log_sigma = torch.clamp(log_sigma, -20, 2)

        return mu, log_sigma

class DuckieNetwork(MultiheadNetwork):
    def __init__(self, in_channels, action_size):
        base_net = BaseNet(in_channels)
        base_out_size = 1024
        critic_1_net = CriticNet(base_out_size, action_size)
        critic_2_net = CriticNet(base_out_size, action_size)
        target_critic_1_net = CriticNet(base_out_size, action_size)
        target_critic_2_net = CriticNet(base_out_size, action_size)
        policy_net = PolicyNet(base_out_size, action_size)
        networks = SACNetworks(base_net, critic_1_net, critic_2_net, target_critic_1_net, target_critic_2_net, policy_net)
        super().__init__(networks)