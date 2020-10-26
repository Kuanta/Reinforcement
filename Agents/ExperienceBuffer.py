'''
Experience Buffer
'''

import numpy as np

class Experience:
    def __init__(self, state, action, reward, next_state):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state


class ExperienceBuffer:
    def __init__(self, size):
        self.buffer = []
        self.size = size

    def add_experience(self, experience: Experience):
        self.buffer.append(experience)
        if len(self.buffer) > self.size:
            self.buffer.pop(0)  # Remove the oldest experience from head

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[idx] for idx in indices]

    def __len__(self):
        return len(self.buffer)