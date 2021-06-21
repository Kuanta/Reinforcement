from Agents.SAC.SACAgent import SACAgentOptions, SACAgent
from Environments.GymEnvironment import GymEnvironment
from Environments.Environment import ContinuousDefinition
from Environments.wrappers import ResizeWrapper, SwapDimensionsWrapper, ImageNormalizeWrapper
import Trainer as trn
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym_duckietown.envs.duckietown_env as duckie
from gym_duckietown.envs.duckietown_env import DuckietownEnv

import gym_duckietown.wrappers as wrappers

from torch.distributions.normal import Normal
import os, datetime

# Define networks

TEST = False

class SACNetwork(nn.Module):
    def __init__(self, in_channels, action_size):
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
        self.policy_mean = nn.Sequential(
            self.policy_base,
            nn.Linear(32, action_size)
        )

        self.policy_sigma = nn.Sequential(
            self.policy_base,
            nn.Linear(32, action_size)
        )

        self.critic_1 = nn.Sequential(
            nn.Linear(state_size+action_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        self.critic_2 = nn.Sequential(
            nn.Linear(state_size+action_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

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
        sigma = self.policy_sigma(features)

        return mean, sigma

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

duckie.logger.disabled = True # Disable log messages from ducki  
env = DuckietownEnv(
    seed = None,
    map_name = "zigzag_dists",
    max_steps = 500001,
    draw_curve = False,
    draw_bbox = False,
    domain_rand = False,
    randomize_maps_on_reset = False,
    accept_start_angle_deg = 4,
    full_transparency = True,
    user_tile_start = None,
    num_tris_distractors = 12,
    enable_leds = False,
)

env = ResizeWrapper(env, 80, 80)
env = SwapDimensionsWrapper(env)
env = ImageNormalizeWrapper(env)
env = GymEnvironment(env)

state_size = env.gym_env.observation_space.shape[0]
act_size = env.gym_env.action_space.shape[0]
action_def = ContinuousDefinition(env.gym_env.action_space.shape, \
    env.gym_env.action_space.high, \
    env.gym_env.action_space.low)

head_network = HeadNet(3)
head_out_size = 1024
value_net = ValueNet(head_network, head_out_size)
target_value_net = ValueNet(head_network, head_out_size)
actor_net = ActorNet(head_network, head_out_size, action_def, 1e-6)
critic_net_1 = CriticNet(head_network, head_out_size, act_size)
critic_net_2 = CriticNet(head_network, head_out_size, act_size)

multihead_net = SACNetwork(3, act_size)
opts = SACAgentOptions()

opts.exp_buffer_size = 100000
opts.learning_rate = 0.0003
opts.exp_batch_size = 256
opts.tau = 0.005
opts.use_gpu = True
opts.clustering = False
opts.cluster_samples = 30000
opts.use_elbow_plot = False
opts.n_clusters = 30
opts.n_episodes_exploring = 150  # Half  of the epiodes with exploring
opts.n_episodes_exploring_least_acts = 150
opts.render = False

agent = SACAgent(multihead_net, actor_net, critic_net_1, critic_net_2, value_net, target_value_net, action_def, opts)
if not TEST:

    time = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    save_path = os.path.join("./duckie_models", time)
    trnOpts = trn.TrainOpts()
    trnOpts.n_epochs = 1
    trnOpts.n_episodes = 5
    trnOpts.save_path = save_path


    trainer = trn.Trainer(agent, env, trnOpts)
    trainer.train()
    agent.save_model(save_path)
else:
    agent.load_model("duckie_models/2021-06-21_19-42-49/multihead")
    total_reward = 0
    state = env.reset()

    while True:
        action = agent.act(state, evaluation=True)
        print(action)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        total_reward += reward
        env.render()
        if done:
            print("Total reward:"+total_reward)
            break