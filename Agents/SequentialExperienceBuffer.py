import numpy as np
from collections import namedtuple
import random
import torch

class Episode:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.done_flags =[]
    
    def add_transition(self, state, action, reward, next_state, done_flag):
        self.states.append(state)
        self.actions.append(torch.tensor(action))
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.done_flags.append(int(done_flag))

class SequentialExperienceBuffer:
    def __init__(self,n_episodes, n_sequence): 
        self.n_episodes = n_episodes
        self.n_sequence = n_sequence
        self.episodes = []

    def is_accumulated(self, size):
        if len(self.episodes) >= size:
            return True
        return False
        
    def add_episode(self, episode: Episode):
        self.episodes.append(episode)
        if len(self.episodes) > self.n_episodes:
            self.episodes.pop(0)
    
    def sample(self, n_episodes, sequence_length, device):
        if len(self.episodes) < n_episodes:
            return None
        episodes = random.sample(self.episodes, n_episodes)
        sampled_states = []
        sampled_actions = []
        sampled_rewards = []
        sampled_next_states = []
        sampled_done_flags = []
        for episode in episodes:
            start = np.random.randint(0, len(episode.states)+1-sequence_length)
            sampled_states.append(torch.stack(episode.states[start:start+sequence_length]))
            sampled_actions.append(torch.stack(episode.actions[start:start+sequence_length]))
            sampled_rewards.append(torch.tensor(episode.rewards[start:start+sequence_length]))
            sampled_next_states.append(torch.stack(episode.next_states[start:start+sequence_length]))
            sampled_done_flags.append(torch.tensor(episode.done_flags[start:start+sequence_length]))
        
        return torch.stack(sampled_states).float().to(device), \
         torch.stack(sampled_actions).float().to(device), \
         torch.stack(sampled_rewards).float().to(device), torch.stack(sampled_next_states).float().to(device), torch.stack(sampled_done_flags).float().to(device)

