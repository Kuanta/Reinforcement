'''
Experience Buffer
'''

import numpy as np
import torch
from sklearn.cluster import KMeans
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def smooth(signal):
    nbins = signal.size
    sigma = 0.25  # Use a small sigma
    kernels = []
    peaks,_ = find_peaks(signal)
    for i in range(nbins):
        kernel = [np.exp(-np.power(i-j, 2)/(2*sigma*sigma))*signal[i] for j in range(nbins)]
        kernels.append(kernel)

    kernels = np.array(kernels, dtype=float)
    kernels = np.transpose(np.transpose(kernels).sum(1))
    return kernels


class Cluster:
    def __init__(self, center, action_def, n_bins=20):
        self.center = center
        self.actions = []
        self.rewards = []
        self.n_bins = n_bins
        self.action_buffer_size = 100  # Number of acts to hold
        # For continuous
        if action_def.continuous:
            self.action_size = action_def.shape[0]
            self.action_bins = np.zeros((self.action_size, n_bins), dtype=float)  # Bins between -1 and 1. Second dimension is more precise
            self.reward_bins = np.zeros((self.action_size, n_bins), dtype=float) 
        else:
            self.action_size = 1
            self.action_bins = np.zeros(1)
            self.reward_bins = np.zeros(1)
        self.n_points = 0
    
    def update_action_bins(self):
        # Add rewards also
        actions = np.transpose(np.array(self.actions, dtype=float))
        for a in range(actions.shape[0]):
            for i in range(actions.shape[1]):
                action = actions[a][i]
                reward = self.rewards[i]
                index = self.get_bin_index(action)
                self.action_bins[a][index] += 1
                self.reward_bins[a][index] += reward
                
    def add_action(self, action, reward):
        '''
        Adds an action to the action list. Actually this may not be necessary because we hold the histogram of actions
        '''
        self.actions.append(action)
        self.rewards.append(reward)
        if len(self.actions) > self.action_buffer_size:
            # Pop from head
            self.actions.pop(0)  
            self.rewards.pop(0)

        # Find the bin of the action
        for a in range(self.action_size):
            act = action[a]
            index = self.get_bin_index(act)
            self.action_bins[a][index] += 1
            self.reward_bins[a][index] += reward

    def get_bin_index(self, action):
        delta = 2/self.n_bins
        index = np.floor((action - (-1))/delta)
        return int(index)
    def add_point(self, point):
        self.n_points += 1
        self.center = self.center + point/self.n_points

    def generate_action(self, act_def):
        '''
        Generates an action using the previously stored actions
        '''
        if act_def.continuous:
            actions = []
            for a in range(self.action_size):
                # Check if empty
                if self.action_bins[a].sum() == 0:
                    index = np.random.randint(0, self.n_bins)
                else:
                    rough_bins = self.action_bins[a]
                    smooth_bins = smooth(rough_bins)
                    index = np.argmin(smooth_bins)

                bin_low = -1 + index*(2/self.n_bins)
                bin_high = -1 + (index+1)*(2/self.n_bins)
                action = np.random.rand(bin_low, bin_high)
                actions.append(action)
            return actions
        else:
            # Discrete action spaces need environment specific knowledge
            # Maybe select the least used action
            return 0
        
    def generate_action_reward(self, act_def):
        '''
        Generates an action using the previously stored actions and rewards. 
        '''
        if act_def.continuous:
            actions = []
            for a in range(self.action_size):
                # Check if empty
                if self.action_bins[a].sum() == 0:
                    index = np.random.randint(0, self.n_bins)
                    
                else:
                    bins = self.reward_bins[a]
                    index = np.argmax(bins)

                bin_low = -1 + index*(2/self.n_bins)
                bin_high = -1 + (index+1)*(2/self.n_bins)
                action = np.random.uniform(bin_low, bin_high)
                actions.append(action)

            return actions
        else:
            # Discrete action spaces need environment specific knowledge
            # Maybe select the least used action
            return 0
class ExperienceBuffer:
    def __init__(self, size, act_def):
        self.size = size
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.done = []
        self.clusters = []  # If clustering is enabled
        self.act_def = act_def

        self.last_cluster_id = -1  # Cluster id of the last classified

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
        states = torch.tensor(states).to(dtype=dtype, device=device)
        actions = torch.tensor(actions).to(dtype=dtype, device=device)
        rewards = torch.tensor(rewards).to(dtype=dtype, device=device)
        next_states = torch.tensor(next_states).to(dtype=dtype, device=device)
        done = torch.tensor(done).to(dtype=dtype, device=device)
        return states, actions, rewards, next_states, done

    def is_accumulated(self, batch_size):
        '''
        Returns true if enough experience is accumulated
        :return: Boolean flag
        '''
        if self.__len__() >= batch_size:
            return True
        else:
            return False
    
    # Clustering Methods
    def cluster(self, n_clusters, use_elbow_plot=False):
        '''
        Cluster states in the replay buffer
        '''
        if use_elbow_plot:
            clusters = range(5,100)  # This can be an option as well
            squared_dists = []
            for i in clusters:
                km = KMeans(n_clusters=i)
                km = km.fit(self.states)
                squared_dists.append(km.inertia_)
            plt.plot(clusters, squared_dists, 'bx-')
            plt.xlabel("K")
            plt.show()
            n_clusters = int(input("Enter the desired cluster count"))
            
        kmeans = KMeans(init="random", n_clusters=n_clusters, n_init=10, max_iter=1000)
        kmeans.fit(self.states)
        for center in kmeans.cluster_centers_:
            self.clusters.append(Cluster(center, self.act_def))

        act_mag = (self.act_def.upper_limit - self.act_def.lower_limit) / 2
        act_bias = (self.act_def.upper_limit + self.act_def.lower_limit) / 2
        # Add existing actions to the clusters
        for i in range(len(self.actions)):
            label = kmeans.labels_[i]
            # Normalize action
            action = self.actions[i] / act_mag + act_bias
            reward = self.rewards[i]
            self.clusters[label].actions.append(action)
            self.clusters[label].rewards.append(reward)
        
        for cluster in self.clusters:
            cluster.update_action_bins()

    def classify(self, state, threshold_scale = 5):
        '''
        Updates the clusters
        '''
        cluster_centers = np.array([cluster.center for cluster in self.clusters])
        distances = cluster_centers - state
        distances = np.power(distances, 2)
        distances = np.sum(distances, 1)
        distances = np.sqrt(distances)
        index = np.argmin(distances)

        cc = cluster_centers - np.expand_dims(cluster_centers, 1)
        cc = np.power(cc, 2)
        cc = np.sum(2)
        cc = np.sqrt(cc)
        avg_dist = np.mean(cc)

        if distances[index] < avg_dist*threshold_scale:  
            self.clusters[index].add_point(state)
        else:
            # This state is too far away, create a new cluster
            new_cluster = Cluster(state, self.act_def)
            self.clusters.append(new_cluster)
            index = len(self.clusters)-1  # index isfrom scipy.optimize import minimize the last
        
        self.last_cluster_id = index
        return index


    def clear(self):
        '''
        Clears the buffer
        '''
        self.buffer = []
        self.clusters = []

    def __len__(self):
        return len(self.states)
