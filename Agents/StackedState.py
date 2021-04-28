from collections import deque
import numpy as np

class StackedState:
    def __init__(self, stack_size):
        self.queue = []
        self.stack_size = stack_size

    def add_state(self, state):
        if self.queue is None or len(self.queue) < 1:
            # Queue is empty, fill it with the same state
            for i in range(self.stack_size):
                self.queue.append(state)

        else:
            self.queue.append(state)
        
        if len(self.queue) > self.stack_size:
            self.queue.pop(0)
            
    def get_stacked_states(self):
        '''
        Turns the current queue into a stacked state. First channel is the latest channel
        '''
        return np.stack(self.queue, axis=0)

    def add_and_get(self, state):
        self.add_state(state)
        return self.get_stacked_states()

    def reset(self):
        # Clear existing queue
        self.queue = []