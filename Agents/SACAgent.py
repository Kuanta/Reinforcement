'''
Soft Actor Critic Agent
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

class SACAgent(Agent):
    def __init__(self, actor_net, critic_net, )