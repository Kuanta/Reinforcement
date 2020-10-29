'''
Experience Buffer
'''

import numpy as np
import torch


class ExperienceBuffer:
    def __init__(self, size):
        self.size = size
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.done = []

    def add_experience(self, curr_state, action, reward, next_state, done):
        self.states.append(curr_state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.done.append(done)
        if len(self.states) > self.size:
            self.states.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            self.next_states.pop(0)
            self.done.pop(0)

    def sample(self, batch_size):
        '''
        Samples a batch of transitions with the given batch size.
        Elements of a transitions are stored in their individual arrays to handle dimension variations among them.
        :param batch_size(int): Number of experience to sample
        :return:
        '''
        indices = np.random.choice(len(self.states), batch_size, replace=False)
        sampled_states = []
        sampled_actions = []
        sampled_rewards = []
        sampled_next_states = []
        sampled_done_flags = []
        for idx in indices:
            sampled_states.append(self.states[idx])
            sampled_actions.append(self.actions[idx])
            sampled_rewards.append(self.rewards[idx])
            sampled_next_states.append(self.next_states[idx])
            sampled_done_flags.append((self.done[idx]))
        return sampled_states, sampled_actions, sampled_rewards, \
            sampled_next_states, sampled_done_flags

    def sample_numpy(self, batch_size):
        '''
        Returns the sampled experiences as numpy arrays
       :param batch_size: Number of experiences in the sample
       :return: Numpy arrays in the form of states, actions, rewards, next_states, done
        '''
        states, actions, rewards, next_states, done = self.sample(batch_size)
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        done = np.array(done)
        return states, actions, rewards, next_states, done

    def sample_tensor(self,batch_size, device, dtype):
        '''
        Returns the sampled experiences as torch tensors
       :param batch_size: Number of experiences in the sample
       :param device: Tensor device
       :param dtype: Data type
       :return: Tensors in the form of states, actions, rewards, next_states, done
        '''
        states, actions, rewards, next_states, done = self.sample(batch_size)
        states = torch.stack(states).to(dtype=dtype, device=device)
        actions = torch.stack(actions).to(dtype=dtype, device=device)
        rewards = torch.stack(rewards).to(dtype=dtype, device=device)
        next_states = torch.stack(next_states).to(dtype=dtype, device=device)
        done = torch.stack(done).to(dtype=dtype, device=device)
        return states, actions, rewards, next_states, done

    def is_accumulated(self):
        '''
        Returns true if enough experience is accumulated
        :return: Boolean flag
        '''
        if self.__len__() >= self.size:
            return True
        else:
            return False

    def clear(self):
        '''
        Clears the buffer
        '''
        self.buffer = []

    def __len__(self):
        return len(self.states)
