'''
Some pre-defined networks for ease of use are defined in this script
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ReinforceNetwork(nn.Module):
    def __init__(self, input_size, n_actions):
        super(ReinforceNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        return F.softmax(self.net(x), dim=1)


class DQNetwork(nn.Module):
    def __init__(self, n_actions, n_channels):
        super(DQNetwork, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            )
        self.fc1 = nn.Linear(2816, 512)
        self.fc2 = nn.Linear(512, n_actions)
    
    def forward(self, x):
        x = self.convs(x)
        x = self.fc1(x.view(x.shape[0], -1))
        x = nn.functional.relu(x)
        x = self.fc2(x)
        return x
        
class ActorCriticNetwork(nn.Module):
    def __init__(self, obs_size, n_actions):
        super(ActorCriticNetwork, self).__init__()
        self.common = nn.Sequential(
            nn.Linear(obs_size, 128),
            nn.ReLU(),
        )
        self.actor_path = nn.Linear(128, n_actions)
        self.critic_path = nn.Linear(128, 1)

    def forward(self, x):

        x = self.common(x)
        actor = F.softmax(self.actor_path(x))
        critic = self.critic_path(x)
        return actor, critic


def fanin_init(size, fanin=None):
    '''
    Initialization explained in the DDQN paper
    :param size:
    :param fanin:
    :return:
    '''
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

class ActorNetwork(nn.Module):
    def __init__(self, n_states, n_actions, device='cpu'):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(n_states, 300)
        self.fc2 = nn.Linear(300, 400)
        self.fc3 = nn.Linear(400, n_actions)
        self.init_weights()
        self.device = device
        self.to(device)
        self.scaling_factor = nn.Parameter(torch.tensor(1).float().to(device))  # Learnable scale factor

    def init_weights(self, init_w=3e-3):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)

    def forward(self, observation):
        x = self.fc1(observation.float())
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = torch.tanh(x)
        x = x * self.scaling_factor
        return x

class CriticNetwork(nn.Module):
    def __init__(self, n_states, n_actions, device='cpu'):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(n_states, 300)
        self.fc2 = nn.Linear(300+n_actions, 400)
        self.fc3 = nn.Linear(400, n_actions)
        self.init_weights()
        self.device = device
        self.to(device)

    def init_weights(self, init_w=3e-3):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)

    def forward(self, observation, actions):
        x = self.fc1(observation.float())
        x = F.relu(x)
        x = self.fc2(torch.cat([x, actions.float()], 1))
        x = F.relu(x)
        x = self.fc3(x)
        return x
