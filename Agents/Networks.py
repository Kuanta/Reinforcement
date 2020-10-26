'''
Some pre-defined networks for ease of use are defined in this script
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

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


class ActorCriticNetwork(nn.Module):
    def __init__(self, obs_size, n_actions):
        super(ActorCriticNetwork, self).__init__()
        self.common = nn.Sequential(
            nn.Linear(obs_size, 128),
            nn.ReLU(),
        )
        self.actor_path = nn.Linear(128,n_actions)
        self.critic_path = nn.Linear(128, 1)

    def forward(self, x):
        x = self.common(x)
        actor = F.softmax(self.actor_path(x))
        critic = self.critic_path(x)
        return actor, critic
