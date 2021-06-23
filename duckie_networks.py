import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from Environments.Environment import ContinuousDefinition


def init_weight(m):
    if type(m) == nn.Linear:
        m.weight.data.uniform_(-3e-3,3e-3)
        m.bias.data.uniform_(-3e-3,3e-3)
 
class SACNetwork(nn.Module):

    def __init__(self, in_channels, action_size, init_w = 3e-3):
        super(SACNetwork, self).__init__()
        self.feature_extractor = nn.Sequential(nn.Conv2d(in_channels, out_channels=32, kernel_size=4, stride=2),
        nn.ReLU(),
        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2),
        nn.ReLU(),
        nn.Conv2d(in_channels=32, out_channels=16, kernel_size=4, stride=2),
        nn.ReLU())
        state_size = 1024
        self.reparam_noise = 1e-6
        self.value = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.value.apply(init_weight)

        self.target_value = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        self.policy_base = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.policy_base.apply(init_weight)

        self.policy_mean = nn.Sequential(
            self.policy_base,
            nn.Linear(32, action_size)
        )
        self.policy_mean.apply(init_weight)

        self.policy_sigma = nn.Sequential(
            self.policy_base,
            nn.Linear(32, action_size)
        )
        self.policy_sigma.apply(init_weight)

        self.critic_1 = nn.Sequential(
            nn.Linear(state_size+action_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.critic_1.apply(init_weight)

        self.critic_2 = nn.Sequential(
            nn.Linear(state_size+action_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.critic_2.apply(init_weight)
        
        # Freeze params of value target
        for param in self.target_value.parameters():
            param.requires_grad = False


    def forward(self, x):
        return self.feature_extraction(x)

    def feature_extraction(self, x):
        features = self.feature_extractor(x).view(x.shape[0], -1)  # Flattened features
        return features

    def get_value(self, features):
        return self.value(features)

    def get_target_value(self, features):
        with torch.no_grad():
            return self.target_value(features)

    def get_critics(self, features, actions):
        x = torch.cat([features,actions],1)
        critic_1 = self.critic_1(x)
        critic_2 = self.critic_2(x)
        return critic_1, critic_2
    
    def get_mean_sigma(self, features):
        mean = self.policy_mean(features)
        log_sigma = self.policy_sigma(features)
        log_sigma = torch.clamp(log_sigma, -20, 2)
        return mean, log_sigma

    def sample(self, features, add_noise=False):
        '''
        From the Appendix C of SAC paper:
        log(pi) = log(Gaussian) - sum(log(1-tanh(ui)^2))
        '''
        mu, log_sigma = self.get_mean_sigma(features)
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

# The remainings doesn't have a use for now 
class HeadNet(nn.Module):
    def __init__(self, in_channels):
        super(HeadNet, self).__init__()
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
    def __init__(self, head_network, n_inputs):
        super(ValueNet, self).__init__()
        self.head = head_network
        self.n_inputs = n_inputs
        self.fc1 = nn.Linear(n_inputs, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
        self.fc3.weight.data.uniform_(-(3e-3), 3e-3)
        self.fc3.bias.data.uniform_(-(3e-3), 3e-3)
    
    def forward(self, states):
        features = self.head(states).view(states.shape[0], -1)
        x = self.fc1(features)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

class CriticNet(nn.Module):
    def __init__(self, head, n_states, n_actions):
        super(CriticNet ,self).__init__()
        self.head = head
        self.fc1 = nn.Linear(n_states + n_actions, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, states, actions):
        features = self.head(states).view(states.shape[0], -1)
        x = torch.cat([features,actions],1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

class ActorNet(nn.Module):
    def __init__(self, head, n_states, act_def: ContinuousDefinition, reparam_noise):
        super(ActorNet, self).__init__()
        self.act_def = act_def
        self.head = head
        self.fc1 = nn.Linear(n_states, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mean_layer = nn.Linear(256, act_def.shape[0])
        self.sigma_layer = nn.Linear(256, act_def.shape[0])
        self.reparam_noise = reparam_noise

    def forward(self, states):
        features = self.head(states).view(states.shape[0], -1)
        x = self.fc1(features)
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
        return normalized_actions, log_probs, torch.tanh(mu)

class HeadNet(nn.Module):
    def __init__(self, in_channels):
        super(HeadNet, self).__init__()
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
    def __init__(self, head_network, n_inputs):
        super(ValueNet, self).__init__()
        self.head = head_network
        self.n_inputs = n_inputs
        self.fc1 = nn.Linear(n_inputs, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
        self.fc3.weight.data.uniform_(-(3e-3), 3e-3)
        self.fc3.bias.data.uniform_(-(3e-3), 3e-3)
    
    def forward(self, states):
        features = self.head(states).view(states.shape[0], -1)
        x = self.fc1(features)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

class CriticNet(nn.Module):
    def __init__(self, head, n_states, n_actions):
        super(CriticNet ,self).__init__()
        self.head = head
        self.fc1 = nn.Linear(n_states + n_actions, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, states, actions):
        features = self.head(states).view(states.shape[0], -1)
        x = torch.cat([features,actions],1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

class ActorNet(nn.Module):
    def __init__(self, head, n_states, act_def: ContinuousDefinition, reparam_noise):
        super(ActorNet, self).__init__()
        self.act_def = act_def
        self.head = head
        self.fc1 = nn.Linear(n_states, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mean_layer = nn.Linear(256, act_def.shape[0])
        self.sigma_layer = nn.Linear(256, act_def.shape[0])
        self.reparam_noise = reparam_noise

    def forward(self, states):
        features = self.head(states).view(states.shape[0], -1)
        x = self.fc1(features)
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