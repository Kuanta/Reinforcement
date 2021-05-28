import gym
import torch
import numpy as np
from Environments.GymEnvironment import GymEnvironment
from Environments.Environment import DiscreteDefinition
from skimage.color import rgb2gray
from skimage import transform
from Agents.Networks import DQNetwork
from Agents.DQAgent import DQAgent, DQAgentOpts
import cv2 as cv

TRAIN = False
TEST = True

def pre_process(state):
    gray = rgb2gray(state)
    org_shape = gray.shape
    cropped = gray[8:-12, 4:-12]
    normalized = cropped
    processed = transform.resize(normalized, org_shape)
    return processed

retro_env = gym.make('SpaceInvaders-v0')
env = GymEnvironment(retro_env, pre_process)
network = DQNetwork(env.gym_env.action_space.n, 4)

# Possible actions
action_samples = np.array([0,1,2,3,4,5])
options = DQAgentOpts()
options.loss = torch.nn.functional.smooth_l1_loss
options.use_exp_stack = True
options.exp_stack_size = 4
options.use_gpu = True
options.target_update_freq = 500
options.epsilon_decay = 100
options.max_epsilon = 1
options.min_epsilon = 0.01
options.render = False
agent = DQAgent(network, DiscreteDefinition(action_samples), opts=options)

if TRAIN:
    agent.load_model("space_invaders_2")
    agent.learn(env, 20, 50000)
    agent.save_model("space_invaders_2")

if TEST:
    agent.load_model("space_invaders_2")
    
    for e in range(10):
        curr_state = env.reset()
        curr_state = agent.exp_stack.add_and_get(curr_state)
        episode_reward = 0
        iteration = 0
        action = 0
        while True:
            if iteration % agent.opts.exp_stack_size == 0:
                action = agent.act_greedy(torch.tensor(curr_state).float())
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            agent.exp_stack.add_state(next_state)
            env.render()
            curr_state = agent.exp_stack.get_stacked_states()
            if done:
                print("Episode reward {}".format(episode_reward))
                break
