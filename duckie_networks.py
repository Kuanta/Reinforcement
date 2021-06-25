from Agents.SAC.Networks import MultiheadNetwork, SACNetworks
import torch
import torch.nn as nn
import torch.nn.functional as F

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class BaseNet(nn.Module):
    def __init__(self, in_channels):
        super(BaseNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, 2, 1),          # B,  32, 32, 32
            nn.ReLU(),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32, 16, 16
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),          # B,  64,  8,  8
            nn.ReLU(),
            nn.Conv2d(64, 64, 4, 2, 1),          # B,  64,  4,  4
            nn.ReLU(),
            nn.Conv2d(64, 256, 4, 1),            # B, 256,  1,  1
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.layers(x).view(x.shape[0], -1)
        return out

class CriticNet(nn.Module):
    def __init__(self, n_states, n_actions, use_cnn=False):
        super(CriticNet ,self).__init__()
        if use_cnn:
            self.base_net = BaseNet(3)
        else:
            self.base_net = None
        self.n_states = n_states
        self.fc1 = nn.Linear(n_states + n_actions, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
        self.fc3.weight.data.uniform_(-(3e-3), 3e-3)
        self.fc3.bias.data.uniform_(-(3e-3), 3e-3)

    def forward(self, states, actions):
        if self.base_net is not None:
            img = states
            x = self.base_net(img)
        else:
            x = states
        x = torch.cat([x, actions], 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

class ActorNet(nn.Module):
    def __init__(self, n_states, action_size, use_cnn=False):
        super(ActorNet, self).__init__()
        if use_cnn:
            self.base_net = BaseNet(3)
        else:
            self.base_net = None
        self.fc1 = nn.Linear(n_states, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mean_layer = nn.Linear(256, action_size)
        self.sigma_layer = nn.Linear(256, action_size)
        self.mean_layer.weight.data.uniform_(-(3e-3), 3e-3)
        self.mean_layer.bias.data.uniform_(-(3e-3), 3e-3)
        self.sigma_layer.weight.data.uniform_(-(3e-3), 3e-3)
        self.sigma_layer.bias.data.uniform_(-(3e-3), 3e-3)

    def forward(self, states):
        if self.base_net is not None:
            img = states
            x = self.base_net(img)
        else:
            x = states
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        
        mu = self.mean_layer(x)
        log_sigma = self.sigma_layer(x)
        log_sigma = torch.clamp(log_sigma, -20, 2)

        return mu, log_sigma

class DuckieNetwork(MultiheadNetwork):
    def __init__(self, state_size, action_size):
        critic_1_net = CriticNet(state_size, action_size)
        critic_2_net = CriticNet(state_size, action_size)
        target_critic_1_net = CriticNet(state_size, action_size)
        target_critic_2_net = CriticNet(state_size, action_size)
        policy_net = ActorNet(state_size, action_size)
        networks = SACNetworks(None, critic_1_net, critic_2_net, target_critic_1_net, target_critic_2_net, policy_net)
        super().__init__(networks)