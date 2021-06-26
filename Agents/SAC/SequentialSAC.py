'''
Soft Actor Critic Agent for RNN
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from Environments.Environment import Environment, DiscreteDefinition, ContinuousDefinition
from Agents.Agent import Agent
from Agents.SAC.SACAgent import SACAgent, SACAgentOptions
from Agents.SAC.Networks import MultiheadNetwork
from Agents.SequentialExperienceBuffer import SequentialExperienceBuffer, Episode
import Agents.ExperienceBuffer as exp
import numpy as np
from Agents.BaseAgentOptions import BaseAgentOptions
import os, json, time
from Trainer import TrainOpts

class SequentialSACAgentOptions(SACAgentOptions):
    def __init__(self):
        super().__init__()
        self.sequence_length = 8


class SequentialSACAgent(SACAgent):
    def __init__(self, multihead_net: MultiheadNetwork, act_def: ContinuousDefinition, 
    opts:SequentialSACAgentOptions=SequentialSACAgentOptions()):
        super(Agent).__init__()  # Initialize the Agent class constructor
        self.opts = opts
        self.act_def = act_def
        self.exp_buffer = SequentialExperienceBuffer(self.opts.exp_buffer_size, self.opts.sequence_length)
        self.multihead_net = multihead_net
        self.multihead_net.init_network(self.opts.learning_rate)
        self.init_entropy()
        if not getattr(self.multihead_net, "sample"):
            raise("Network must implement 'sample' method")

    def act(self, new_state, hidden_states, device, evaluation=False):
        # Base network must be responsible of returning the features
        # Unsqueeze new_state so that it represents a sequence
        features, hidden_states = self.multihead_net.feature_extraction(new_state.unsqueeze(0), hidden_states) 
        # TODO: For now evaluation and no evaluation same. 
        if not evaluation:
            with torch.no_grad():
                action, _, _, hidden_states  = self.multihead_net.sample(features, hidden_states, add_noise=True)
            return action.squeeze(0).squeeze(0).detach().cpu().numpy(), hidden_states
        else:
            with torch.no_grad():
                action , _, _, hidden_states  = self.multihead_net.sample(features, hidden_states, add_noise=False)
            return action.squeeze(0).squeeze(0).detach().cpu().numpy(), hidden_states
    
    def update_params(self, n_iter, device):

        # Learn if enough data is accumulated
        if self.exp_buffer.is_accumulated(self.opts.exp_batch_size):
            if(self.opts.clustering and len(self.exp_buffer.clusters) == 0):
                return

            # Sample from buffer
            s_states, s_actions, s_rewards, s_next_states, s_done =\
            self.exp_buffer.sample(self.opts.exp_batch_size, self.opts.sequence_length, device)

            features, hx = self.multihead_net(s_states, None)  # Not passing hidden states means the network should initialzie it as 0s

            # Target Values
            with torch.no_grad():
                target_features, _ = self.multihead_net(s_next_states, hx)
                next_actions, log_probs, _, _ = self.multihead_net.sample(target_features, hx, add_noise=True)
                critic_1_target, critic_2_target = self.multihead_net.get_target_critics(target_features, next_actions, hx)
                critic_target = torch.min(critic_1_target, critic_2_target)
                target_value = critic_target - self.entropy.exp()*log_probs
                target_value = (target_value*(1-s_done.unsqueeze(-1)))
                q_hat = s_rewards.unsqueeze(-1) + self.opts.discount*target_value

            # Tune entropy
            if self.opts.auto_entropy:
                self.entropy_optimizer.zero_grad()
                target_entropy = -1*self.act_def.shape[0]
                entropy_loss = -(self.entropy*(log_probs + target_entropy).detach()).mean()        
                entropy_loss.backward()
                self.entropy_optimizer.step()
            # Optimize Critic
            self.multihead_net.critic_optimizer.zero_grad()
            critic_1, critic_2 = self.multihead_net.get_critics(features, s_actions, hx)
            critic_loss_1 = F.mse_loss(critic_1, q_hat.detach())
            critic_loss_2 = F.mse_loss(critic_2, q_hat.detach())
            critic_loss = critic_loss_1 + critic_loss_2
            critic_loss.backward(retain_graph = True)
            self.multihead_net.critic_optimizer.step()

            # Optimize Policy
            self.multihead_net.policy_optimizer.zero_grad()
            # Calculate critic values for value and policy using the actions sampled from the current policy
            actions, log_probs, _, _ = self.multihead_net.sample(features, hx, add_noise=True)
            critic_1_curr, critic_2_curr = self.multihead_net.get_critics(features, actions, hx)
            critic_curr = torch.min(critic_1_curr, critic_2_curr)
            actor_loss = (self.entropy.exp()*log_probs - critic_curr).mean()
            actor_loss.backward()
            self.multihead_net.policy_optimizer.step()

            if self.multihead_net.base_net is not None:
                self.multihead_net.base_optimizer.step()

            if n_iter % 1 == 0:
                self.multihead_net.update_targets(self.opts.tau)
        
    def learn(self, env:Environment, trnOpts: TrainOpts):
        device = "cpu"
        if self.opts.use_gpu and torch.cuda.is_available():
            device = "cuda:0"
        
        # Load checkpoint
        all_rewards, avg_rewards = self.load_checkpoint(trnOpts.checkpoint)
        self.multihead_net.to(device)
        n_iter = 0
        e = 0
        max_episodes = trnOpts.n_episodes
        max_steps = trnOpts.n_iterations
        while e < max_episodes:  # Looping episodes
            if max_steps > 0 and n_iter > max_steps:
                break
            curr_state = env.reset()
            curr_state = torch.from_numpy(curr_state).to(device).float().unsqueeze(0)
            episode_rewards = []
            step = 0
            episode = Episode()  # Each episode starts with fresh
            hidden_state = None
            while True:
                n_iter += 1
                step += 1
                # Collect experience
                # e < self.opts.n_episodes_exploring => This can be added too
                
                with torch.no_grad():
                    
                    action, hidden_state = self.act(curr_state, hidden_state, device)
                    next_state, reward, done, _ = env.step(action)
                    if self.opts.render:
                        env.render()
                    episode_rewards.append(reward)
                
                next_state = torch.from_numpy(next_state).to(device).float().unsqueeze(0)
                episode.add_transition(curr_state.squeeze(0).detach().cpu().data, action, reward, next_state.squeeze(0).detach().cpu().data, done)    
                if done:
                    if len(episode.states) > self.opts.sequence_length:
                        self.exp_buffer.add_episode(episode)
                        
                    if not self.exp_buffer.is_accumulated(self.opts.exp_batch_size):
                        print("Accumulating buffer iteration: {}".format(n_iter))
                    
                    else:
                        episode_end_reward = np.array(episode_rewards).sum()
                        all_rewards.append(episode_end_reward)
                        e += 1  # Update episode
                        avg_reward = np.mean(all_rewards[-100:])
                        avg_rewards.append(avg_reward)
                        print("({}/{}) - End of episode with total reward: {} - Avg Reward: {} Total Iter: {}".format(e, max_episodes, episode_end_reward, avg_reward, step))
                    break
                
                curr_state = next_state

                # Learn if enough data is accumulated
                if self.exp_buffer.is_accumulated(self.opts.exp_batch_size):
                    # start = time.time()
                    self.update_params(n_iter, device)
                    # end = time.time()
                    # print("Elapsed :{}".format(end-start))
                    
                if n_iter > 0 and self.opts.save_frequency > 0 and n_iter % self.opts.save_frequency == 0:
                    print("Saving at iteration {}".format(n_iter))
                    path = os.path.join(trnOpts.save_path, time.strftime("%Y%m%d-%H%M%S"))

                    self.save_model(path)
                    self.save_rewards(path, all_rewards, avg_rewards)
        
        
        return all_rewards, avg_rewards
