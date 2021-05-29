'''
Soft Actor Critic Agent
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from Environments.Environment import Environment, DiscreteDefinition, ContinuousDefinition
from Agents.Agent import Agent
from Agents.ExperienceBuffer import ExperienceBuffer
from Agents.StackedState import StackedState
from common import polyak_update, freeze_network
import Agents.ExperienceBuffer as exp
import numpy as np
import random
import copy
from Agents.BaseAgentOptions import BaseAgentOptions

class SACAgentOptions(BaseAgentOptions):
    def __init__(self):
        super().__init__()
        self.actor_optimizer = optim.Adam
        self.value_optimizer = optim.Adam
        self.critic_optimizer = optim.Adam
        self.tau = 0.005  # Target net will be updated with this tau at every update call

class SACAgent(Agent):
    '''
    Implements a Soft Actor-Critic agent
    '''
    def __init__(self, actor_net, critic_net_1, critic_net_2, value_net, act_def: ContinuousDefinition, opts:SACAgentOptions=SACAgentOptions()):
        super().__init__()
        self.actor_net = actor_net
        self.critic_net_1 = critic_net_1
        self.critic_net_2 = critic_net_2
        self.value_net = value_net
        self.target_value_net = copy.deepcopy(value_net)
        self.opts = opts
        self.act_def = act_def
        self.actor_optimizer = self.opts.actor_optimizer(self.actor_net.parameters(), self.opts.learning_rate)
        self.critic_optimizer_1 = self.opts.critic_optimizer(self.critic_optimizer_1.parameters(), self.opts.learning_rate)
        self.critic_optimizer_2 = self.opts.critic_optimizer(self.critic_optimizer_2.parameters(), self.opts.learning_rate)
        self.value_optimizer = self.opts.value_optimizer(self.value_net.parameters(), self.opts.learning_rate)
        self.experience_buffer = ExperienceBuffer(self.opts.exp_buffer_size)
        
        if not getattr(self.actor_net, "sample"):
            raise("Actor Network must implement 'sample' method")

        polyak_update(self.target_value_net, self.value_net, 1)

        # Freeze Target Value Parameters
        freeze_network(self.target_value_net)

    def act(self, new_state, device="cpu"):
        action, _ = self.actor_net.sample(new_state)
        return action

    def learn(self, env:Environment, max_episodes:int, max_steps:int):
        device = "cpu"
        if self.opts.use_gpu and torch.cuda.is_available():
            device = "cuda:0"
        
        self.actor_net.to(device)
        self.value_net.to(device)
        avg_rewards = []
        for e in range(max_episodes):
            curr_state = env.reset()
            episode_rewards = []
            for s in range(max_steps):

                # Collect experience
                with torch.no_grad():
                    action = self.act(curr_state)
                    next_state, reward, done, _ = env.step(action)
                    avg_rewards.append(reward)
                    self.experience_buffer.add_experience(curr_state, action, reward, next_state, done)
                
                if done:
                    avg_rewards.append(np.array(episode_rewards).sum())
                    break
                
                if self.experience_buffer.is_accumulated():
                    
                    # Sample from buffer
                    s_states, s_actions, s_rewards, s_next_states, s_done =\
                    self.exp_buffer.sample_numpy(self.opts.exp_batch_size)

                    # Transfer samples to torch tensors
                    s_states = torch.from_numpy(s_states).to(device).float()
                    s_actions = torch.from_numpy(s_actions).to(device).float()  # Continuous actions
                    s_next_states = torch.from_numpy(s_next_states).to(device).float()
                    s_done = torch.from_numpy(s_done).to(device).float()
                    s_rewards = torch.from_numpy(s_rewards).to(device).float()

                    # Optimize Value network
                    self.value_optimizer.zero_grad()
                    actions, log_probs = self.actor_net.sample(s_states, reparameterize=False)
                    critic_1 = self.critic_net_1(s_states, actions)
                    critic_2 = self.critic_net_2(s_states, actions)
                    critic = torch.min(critic_1, critic_2)
                    value_target = critic - log_probs
                    value = self.value_net(s_states)
                    value_loss = 0.5 * F.mse_loss(value_target, value)
                    value_loss.backward(retain_graph=True)  # Why retain graph??
                    self.value_optimizer.step()

                    # Optimize Actor network
                    self.actor_optimizer.zero_grad()
                    actions, log_probs = self.actor_net.sample(s_states, reparameterize=True)  # Sample with noise
                    critic_1 = self.critic_net_1(s_states, actions)
                    critic_2 = self.critic_net_2(s_states, actions)
                    critic = torch.min(critic_1, critic_2)
                    actor_loss = F.mean(critic-log_probs)
                    actor_loss.backward(retain_graph = True)
                    self.actor_optimizer.step()

                    # Optimize Critic Networks
                    self.critic_optimizer_1.zero_grad()
                    self.critic_optimizer_2.zero_grad()
                    with torch.no_grad():
                        target_value = self.target_value_net(s_next_states)*(1-s_done.view(-1,1))  # Check terminal state
                        q_hat = s_rewards + self.opts.discount*target_value
                    q_val_1 = self.critic_net_1(s_states, s_actions)
                    q_val_2 = self.critic_net_2(s_states, s_actions)
                    critic_loss_1 = 0.5*F.mse_loss(q_val_1, q_hat)
                    critic_loss_2 = 0.5*F.mse_loss(q_val_2, q_hat)
                    critic_loss = critic_loss_1 + critic_loss_2
                    critic_loss.backward()
                    self.critic_optimizer_1.step()
                    self.critic_optimizer_2.step()






    def reset():
        pass

