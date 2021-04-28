'''
Deep Q-Learning Agent
'''
import torch
import torch.nn as nn
import torch.optim as optim
from Environments.Environment import Environment, DiscreteDefinition
from Agents.Agent import Agent
from Agents.ExperienceBuffer import ExperienceBuffer
from Agents.StackedState import StackedState
from common import polyak_update
import Agents.ExperienceBuffer as exp
import numpy as np
import random
import copy


class DQAgentOpts(Agent):
    def __init__(self):
        self.exp_batch_size = 64
        self.exp_buffer_size = 10000
        self.use_exp_stack = False
        self.exp_stack_size = 4  # This is the max num for stacked experiences
        self.max_epsilon = 1.0
        self.min_epsilon = 0.1
        self.epsilon_decay = 200  # At each learning step, this value will be multiplied with the epsilon rate
        self.optimizer = optim.Adam
        self.use_gpu = False
        self.learning_rate = 0.0002
        self.discount = 0.99
        self.target_update_freq = 1000  # Update target network every x iteration
        self.verbose = True
        self.verbose_frequency = 1000
        self.render = False

class DQAgent(Agent):
    def __init__(self, network, act_def: DiscreteDefinition, opts=DQAgentOpts()):
        super().__init__()
        self.network = network
        self.target_network = copy.deepcopy(network)
        polyak_update(self.target_network, self.network, 1)
        
        # Freeze target
        for p in self.target_network.parameters():
            p.requires_grad = False

        self.opts = opts
        self.act_def = act_def
        self.exp_buffer = ExperienceBuffer(self.opts.exp_buffer_size)
        self.exp_stack = StackedState(self.opts.exp_stack_size)
        self.epsilon = 1.0


    def act(self, new_state, device="cpu"):
        with torch.no_grad():
            if np.random.random() < self.epsilon:
                action = random.randint(0, len(self.act_def.samples)-1)
            else:
                action = np.argmax(self.network(torch.tensor(new_state).float().to(device).unsqueeze(0)).detach().cpu().numpy())
        return action
        
    def act_greedy(self, new_state, device="cpu"):
        with torch.no_grad():
            net_out = self.network(new_state.unsqueeze(0))
            action = np.argmax(net_out.detach().cpu().numpy())
        return action

    def learn(self, env: Environment, max_episodes: int, max_steps: int):
        device = "cpu"
        if self.opts.use_gpu and torch.cuda.is_available():
            device = "cuda:0"
        print(device)
        self.network.to(device)
        self.target_network.to(device)
        self.reset()
        total_steps = 0
        optimizer = self.opts.optimizer(self.network.parameters(), self.opts.learning_rate)
        avg_rewards = []
        losses = []
        learning_complete = False
        episodes_passed = 0
        while not learning_complete:
            current_step = 0
            target_update_iter = 0
            episode_rewards = []
            curr_state = env.reset()
            action = 0
            if self.opts.use_exp_stack:
                curr_state = self.exp_stack.add_and_get(curr_state)
            #curr_state = torch.tensor(curr_state).to(device).float()
            if episodes_passed > max_episodes:
                    learning_complete = True
                    break
            while True:
                done = 0
                with torch.no_grad():  # Just collecting experience
                    for i in range(self.opts.exp_stack_size-1):
                        action = self.act(curr_state, device)
                    next_state, reward, done, _ = env.step(self.act_def[action])
                    self.exp_stack.add_state(next_state)
                    total_steps += 1 # Doesn't reset
                    next_state = self.exp_stack.get_stacked_states()
                    episode_rewards.append(reward)
                    self.exp_buffer.add_experience(curr_state, action, reward, next_state, done)
                    curr_state = next_state
                    if self.opts.render:
                        env.render()
        
                if done or current_step > max_steps:
                    self.reset()
                    total_episode_reward = np.array(episode_rewards).sum()
                    avg_rewards.append(total_episode_reward)
                    print("({}/{}) - End of episode with total reward: {} iteration: {} Memory Size: {}".format(episodes_passed, max_episodes, total_episode_reward, current_step, len(self.exp_buffer)))
                    break
                
                if self.exp_buffer.is_accumulated():
                    s_states, s_actions, s_rewards, s_next_states, s_done =\
                    self.exp_buffer.sample_numpy(self.opts.exp_batch_size)

                    # TODO: n-step Q-learning
                    optimizer.zero_grad()
                    with torch.no_grad():
                        s_next_states = torch.from_numpy(s_next_states).to(device).float()
                        s_done = torch.from_numpy(s_done).to(device).float()
                        s_rewards = torch.from_numpy(s_rewards).to(device).float()
                        next_state_vals = self.target_network(s_next_states)*(1-s_done.view(-1,1))  # Terminal states has V(s) = 0. That is why we use s_done
                        next_state_vals = next_state_vals*self.opts.discount  # Discount the reward
                        td_target = s_rewards + next_state_vals.max(1)[0].detach()  # In TD target, use target network (see Double Q learning)

                    #loss = -self.opts.loss(td_target, self.network(s_states))
                    s_states = torch.from_numpy(s_states).to(device).float()
                    s_actions = torch.from_numpy(s_actions).to(device).to(torch.int64)
                    curr_state_estimations = self.network(s_states).gather(1, s_actions.view(-1,1))
                    loss = torch.nn.functional.mse_loss(curr_state_estimations, td_target.unsqueeze(1))
                    loss.backward()
                    optimizer.step()

                    target_update_iter += 1
                    
                    losses.append(loss.item())
                    # Update target network
                    if target_update_iter > self.opts.target_update_freq:
                        target_update_iter = 0
                        polyak_update(self.target_network, self.network, 1)
                        print("Update target at step {}".format(total_steps))
                    
                if self.opts.verbose and total_steps%self.opts.verbose_frequency == 0 and len(losses) > 0:
                    print("Total Steps:{} - Loss:{} - Curr Epsilon:{}".format(total_steps, losses[-1], self.epsilon))
                current_step += 1  # Resets every episode
                
                
            
            if self.exp_buffer.is_accumulated():
                episodes_passed += 1  # Increment episode only if enough experience is collected
                
            self.epsilon = self.opts.min_epsilon + (self.opts.max_epsilon - self.opts.min_epsilon)*np.exp(-1.0*episodes_passed/self.opts.epsilon_decay) 

        return avg_rewards, losses

    def save_model(self, PATH):
        torch.save(self.network.state_dict(), PATH)

    def load_model(self, PATH):
        self.network.load_state_dict(torch.load(PATH))

    def reset(self):
        self.exp_stack.reset()
        #self.epsilon = self.opts.max_epsilon
                



