'''
Deep Deterministic Policy Gradient Agent
'''

import torch
import torch.nn as nn
import torch.optim as optim
from Agents.Agent import Agent
from Environments.Environment import Environment
from Agents.ExperienceBuffer import ExperienceBuffer
from Agents.RandomProcesses import OrnsteinUhlenbeckProcess
from common import *
import copy
import math
import numpy as np


class DDPGAgentOptions:
    def __init__(self):
        self.exp_batch_size = 10
        self.exp_buffer_size = 5000
        self.discount = 0.99
        self.actor_optimizer = optim.Adam
        self.critic_optimizer = optim.Adam
        self.actor_learning_rate = 0.001
        self.critic_learning_rate = 0.001
        self.noise_var = 1
        self.noise_epsilon = 5000
        self.noise_depsilon = 1
        self.min_noise_var = 0.01  # If noise variance is smaller than this, no more noise will be added
        self.target_update_delay = 100  # After this many updates in the source networks, update target networks also
        self.act_limit_upper = 2
        self.act_limit_lower = -2
        self.action_scale = 1
        self.action_bias = 0


class DDPGAGent(Agent):
    def __init__(self, actor_network, critic_network, opts=DDPGAgentOptions()):
        super().__init__()
        self.opts = opts
        self.actor_network = actor_network
        self.critic_network = critic_network
        self.target_actor_network = copy.deepcopy(actor_network)
        self.target_critic_network = copy.deepcopy(critic_network)
        # Freeze target networks. They will never be updated with gradient descent/ascent
        for p in self.target_critic_network.parameters():
            p.requires_grad = False
        for p in self.target_actor_network.parameters():
            p.requires_grad = False

        # Init target networks with the same parameters as the source networks
        polyak_update(self.target_actor_network, self.actor_network, 1)
        polyak_update(self.target_critic_network, self.critic_network, 1)
        self.exp_buffer = ExperienceBuffer(self.opts.exp_buffer_size)

        # Initialize optimizer
        self.actor_optimizer = self.opts.actor_optimizer(self.actor_network.parameters(), self.opts.actor_learning_rate)
        self.critic_optimizer = self.opts.critic_optimizer(self.critic_network.parameters(), self.opts.critic_learning_rate)
        self.random_process = OrnsteinUhlenbeckProcess(size=2, theta=0.15, mu=0.0, sigma=0.1)


    def act(self, state, add_noise=False):
        '''
        Generates an action using the actor network. During the network, add some noise to ensure exploration.
        Addition of a noise is the off-policy nature of DDPG
        :param state:
        :param add_noise:
        :return:
        '''

        if type(state) is not torch.Tensor:
            state = torch.tensor(state).to(self.actor_network.device).float()

        bias = torch.tensor(self.opts.action_bias).to(self.actor_network.device).float()
        bias.requires_grad = False
        action = self.actor_network.forward(state)*self.opts.action_scale + bias
        if add_noise:
            if self.opts.noise_epsilon > 0:
                #action = torch.add(action, torch.tensor(self.random_process.sample()).float().to(self.actor_network.device))
                noise = torch.randn(action.shape, dtype=torch.float32) * math.sqrt(self.opts.noise_var)
                action = action + noise.to(self.actor_network.device).float()

        # TODO: Action can be saturated here
        #action = action.clamp(self.opts.act_limit_lower, self.opts.act_limit_upper)
        return action

    def learn(self, environment: Environment, n_episodes: int):
        avg_rewards = []
        for i in range(n_episodes):
            n_update_iter = 0  # Number of update iterations done. Needed to check if target networks need update
            curr_state = torch.tensor(environment.reset()).to(device=self.actor_network.device).float()
            episode_rewards = []
            while True:
                n_update_iter += 1
                action = self.act(curr_state, add_noise=True).cpu().detach().numpy()
                next_state, reward, done = environment.step(action)
                episode_rewards.append(reward)
                self.exp_buffer.add_experience(curr_state, torch.tensor(action).float(), torch.tensor(reward).float(),
                                               torch.tensor(next_state).float(), torch.tensor(done))
                curr_state = torch.tensor(next_state).float().to(self.actor_network.device)
                curr_state.requires_grad = False
                self.opts.noise_epsilon = self.opts.noise_epsilon - self.opts.noise_depsilon
                if done:
                    self.reset()
                    total_episode_reward = np.array(episode_rewards).sum()
                    avg_rewards.append(total_episode_reward)
                    print("({}/{}) - End of episode with total reward: {} iteration: {}".format(i, n_episodes, total_episode_reward, n_update_iter))
                    break
                if self.exp_buffer.is_accumulated():  # Do the updates
                    # Sample experiences
                    #self.critic_network.eval()
                    s_states, s_actions, s_rewards, s_next_states, s_done =\
                        self.exp_buffer.sample_tensor(self.opts.exp_batch_size, device=self.actor_network.device, dtype=torch.float32)

                    critic = self.critic_network.forward(s_states, s_actions.detach())
                    target_actions = self.target_actor_network.forward(s_next_states)
                    target_critics = self.target_critic_network.forward(s_next_states, target_actions)
                    target = s_rewards.view(-1, 1) + self.opts.discount*(1-s_done.view(-1,1))*target_critics

                    # Run Gradient Descent on critic network
                    self.critic_optimizer.zero_grad()
                    #self.critic_network.train()  # Enable train mode
                    critic_loss = torch.nn.functional.mse_loss(critic, target)
                    critic_loss.backward()
                    self.critic_optimizer.step()

                    # Run Gradient Ascent on actor network
                    self.actor_optimizer.zero_grad()
                    self.actor_network.train()  # Enable train mode
                    actor_out = self.act(s_states)
                    actor_loss = -self.critic_network(s_states.detach(), actor_out)
                    actor_loss = actor_loss.mean()
                    actor_loss.backward()
                    self.actor_optimizer.step()
                    self.update_target_networks(0.01)

        return avg_rewards

    def reset(self):
        self.random_process.reset_states()

    def save_model(self, PATH):
        torch.save(self.actor_network.state_dict(), PATH+"_actor")
        torch.save(self.critic_network.state_dict(), PATH+"_critic")

    def load_model(self, PATH):
        self.actor_network.load_state_dict(torch.load(PATH+"_actor"))
        self.critic_network.load_state_dict(torch.load(PATH + "_critic"))

    def update_target_networks(self, p):
        polyak_update(self.target_actor_network, self.actor_network, p)
        polyak_update(self.target_critic_network, self.critic_network, p)
