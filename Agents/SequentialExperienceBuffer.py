import numpy as np
from collections import namedtuple

Episode = namedtuple('Episode', 'state action reward done next_state')

class SequentialExperienceBuffer:
    def __init__(self,size):
        self.size = size
        self.episodes = []
    
    def add_episode(self, episode: Episode):
        self.episodes.append(episode)