'''
Soft Actor Critic Agent
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from Environments.Environment import Environment, DiscreteDefinition, ContinuousDefinition
from Agents.Agent import Agent
from Agents.SAC.Networks import MultiheadNetwork
from Agents.ExperienceBuffer import ExperienceBuffer
from Agents.StackedState import StackedState
from common import polyak_update, freeze_network
import Agents.ExperienceBuffer as exp
import numpy as np
import random
import copy
from Agents.BaseAgentOptions import BaseAgentOptions
import os, time
from Trainer import TrainOpts

class SACAgentOptions(BaseAgentOptions):
    def __init__(self):
        super().__init__()
        self.actor_optimizer = optim.Adam
        self.value_optimizer = optim.Adam
        self.critic_optimizer = optim.Adam
        self.tau = 0.005  # Target net will be updated with this tau at every update call
        self.cluster_samples = 10000  # Number of samples to collect before clustering
        self.clustering = True  # If set to true, states will be clustered
        self.n_clusters = 20
        self.update_cluster_scale = 10  # Check update_cluster func
        self.use_elbow_plot = False
        self.n_episodes_exploring = 100  # Number of episodes using clustered exploration
        self.n_episodes_exploring_least_acts = 50
        self.entropy_scale = 0.5
        self.samples_to_collect_clustering = 100000  # Instead of using clusterd 
        self.cluster_only_for_buffer = False   # If set to true, clustered eexploration will only be used for replay buffer filling

class SACAgent(Agent):
    '''
    Implements a Soft Actor-Critic agent
    '''
    def __init__(self, multihead_net: MultiheadNetwork, act_def: ContinuousDefinition, opts:SACAgentOptions=SACAgentOptions()):
        super().__init__()
        self.opts = opts
        self.act_def = act_def
        self.exp_buffer = ExperienceBuffer(self.opts.exp_buffer_size, act_def)
        self.multihead_net = multihead_net
        self.multihead_net.init_network(self.opts.learning_rate)
        
        if not getattr(self.multihead_net, "sample"):
            raise("Network must implement 'sample' method")

    
    def act_cluster(self, new_state, n_episode):
        # Classify new_state
            index = self.exp_buffer.classify(new_state, self.opts.update_cluster_scale)
            # Get the action of the belonging cluster 
            cluster = self.exp_buffer.clusters[index]
            # Calculate next action
            if n_episode < self.opts.n_episodes_exploring_least_acts:
            # Exploring with using least used actions
                action = cluster.generate_action(self.act_def)
            else:
                # Using acitons with better rewards
                action = cluster.generate_action_reward(self.act_def)
            self.exp_buffer.last_cluster_id = index  # Just in case
            return action

    def act(self, new_state, evaluation=False):
        # TODO: For now evaluation and no evaluation same. 
        if not evaluation:
            #new_state = torch.from_numpy(new_state).to(device).float().unsqueeze(0)
            with torch.no_grad():
                action, _, _  = self.multihead_net.sample(new_state, add_noise=True)
            return action.squeeze(0).detach().cpu().numpy()
        else:
            #new_state = torch.from_numpy(new_state).to(device).float().unsqueeze(0)
            with torch.no_grad():
                action , _, _  = self.multihead_net.sample(new_state, add_noise=False)
            return action.squeeze(0).detach().cpu().numpy()

    def update_params(self, n_iter, device):
        '''
        If multiheaded network, implement this
        '''
        if self.multihead_net is None:
            return 

        # Learn if enough data is accumulated
        if self.exp_buffer.is_accumulated(self.opts.exp_batch_size):
            if(self.opts.clustering and len(self.exp_buffer.clusters) == 0):
                return

            # Sample from buffer
            s_states, s_actions, s_rewards, s_next_states, s_done =\
            self.exp_buffer.sample_tensor(self.opts.exp_batch_size, device, torch.float)

            features = self.multihead_net(s_states)
            target_features = self.multihead_net(s_next_states)

            next_actions, log_probs, _ = self.multihead_net.sample(features, add_noise=True)

            # Target Values
            target_value = self.multihead_net.get_target_value(target_features)
            target_value = (target_value*(1-s_done.view(-1,1)))

            value = self.multihead_net.get_value(features)
            
            
            self.multihead_net.critic_1_optimizer.zero_grad()
            self.multihead_net.critic_2_optimizer.zero_grad()
            # Critics Loss
            with torch.no_grad():
                q_hat = s_rewards.view(-1,1) + self.opts.discount*target_value

            critic_1, critic_2 = self.multihead_net.get_critics(features, s_actions)
            critic_loss_1 = F.mse_loss(critic_1, q_hat.detach())
            critic_loss_2 = F.mse_loss(critic_2, q_hat.detach())
            critic_loss = critic_loss_1 + critic_loss_2
            critic_loss.backward(retain_graph = True)
            self.multihead_net.critic_1_optimizer.step()
            self.multihead_net.critic_2_optimizer.step()

            # Calculate critic values for value and policy using the actions sampled from the current policy
            critic_1_curr, critic_2_curr = self.multihead_net.get_critics(features, next_actions)
            critic_curr = torch.min(critic_1_curr, critic_2_curr)
            # Value Loss
            self.multihead_net.value_optimizer.zero_grad()
            value_target =  (critic_curr - log_probs)
            value_loss = F.mse_loss(value, value_target.detach())
            value_loss.backward(retain_graph=True)
            self.multihead_net.value_optimizer.step()


            # Actor Loss
            self.multihead_net.policy_optimizer.zero_grad()
            actor_loss = torch.mean(log_probs - critic_curr)
            actor_loss.backward(retain_graph=True)
            self.multihead_net.policy_optimizer.step()

            if self.multihead_net.base_net is not None:
                self.multihead_net.base_optimizer.step()

            if n_iter % 2500 == 0:
                print("Critic Loss 1: {}  - Critic Loss 2: {}  Value Loss: {} - Actor Loss: {}".format(critic_loss_1.item(), critic_loss_2.item(),
                value_loss.item(), actor_loss.item()))

            


            if n_iter % 1 == 0:
                polyak_update(self.multihead_net.target_value_net, self.multihead_net.value_net, self.opts.tau)

    def learn(self, env:Environment, trnOpts: TrainOpts):
        device = "cpu"
        if self.opts.use_gpu and torch.cuda.is_available():
            device = "cuda:0"
        
        self.multihead_net.to(device)
        all_rewards = []
        avg_rewards = []
        n_iter = 0
        e = 0
        # explore_episodes = self.opts.n_episodes_exploring
        # if not self.opts.clustering:
        #     explore_episodes = 0

        max_episodes = trnOpts.n_episodes
        max_steps = trnOpts.n_iterations
        while e < max_episodes:  # Looping episodes
            if max_steps > 0 and n_iter > max_steps:
                break
            curr_state = env.reset()
            curr_state = torch.from_numpy(curr_state).to(device).float().unsqueeze(0)
            episode_rewards = []
            step = 0
            while True:
                n_iter += 1
                step += 1
                # Collect experience
                # e < self.opts.n_episodes_exploring => This can be added too
                clustering = self.opts.clustering and e < self.opts.n_episodes_exploring and len(self.exp_buffer.clusters) > 0  # Cluster count being higher than 0 means that clustering has been done
                
                with torch.no_grad():
                    if clustering:
                        action = self.act_cluster(curr_state, e)
                    else:
                        action = self.act(curr_state, device)
                    next_state, reward, done, _ = env.step(action)
                    if self.opts.render:
                        env.render()
                    episode_rewards.append(reward)

                    if clustering:
                        self.exp_buffer.clusters[self.exp_buffer.last_cluster_id].add_action(action, reward)
                      
                    # Check Clustering
                    if self.opts.clustering  and len(self.exp_buffer) > self.opts.cluster_samples \
                    and len(self.exp_buffer.clusters) == 0:  # It means that clustering already done
                        print("Clustering")
                        self.exp_buffer.cluster(self.opts.n_clusters, self.opts.use_elbow_plot)

                
                next_state = torch.from_numpy(next_state).to(device).float().unsqueeze(0)
                self.exp_buffer.add_experience(curr_state.squeeze(0).cpu(), action, reward, next_state.squeeze(0).cpu(), done)   
    
                if done:
                   
                    if not self.exp_buffer.is_accumulated(self.opts.exp_batch_size) or (self.opts.clustering and len(self.exp_buffer.states) < self.opts.cluster_samples):
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
                    #self.update_params(n_iter, device)
                    #start = time.time()
                    self.update_params(n_iter, device)
                    # end = time.time()
                    # print("Elapsed :{}".format(end-start))
                    
                if n_iter > 0 and self.opts.save_frequency > 0 and n_iter % self.opts.save_frequency == 0:
                    print("Saving at iteration {}".format(n_iter))
                    self.save_model(trnOpts.save_path)
        
        
        return all_rewards, avg_rewards

    def save_model(self, PATH):
        # Check Path
        if not os.path.exists(PATH):
            os.mkdir(PATH)
        torch.save(self.multihead_net.state_dict(), os.path.join(PATH, "multihead"))
    
    def load_model(self, PATH):
        self.multihead_net.load_state_dict(torch.load(os.path.join(PATH,"multihead")))

    def reset():
        pass

