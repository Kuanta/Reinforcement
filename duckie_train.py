from torch.serialization import save
from Agents.SAC.SACAgent import SACAgentOptions, SACAgent
from Environments.GymEnvironment import GymEnvironment
from Environments.Environment import ContinuousDefinition
from Environments.wrappers import ResizeWrapper, SwapDimensionsWrapper, ImageNormalizeWrapper
import Trainer as trn
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym_duckietown.envs.duckietown_env as duckie
from gym_duckietown.envs.duckietown_env import DuckietownEnv
import os, datetime, argparse
from duckie_networks import *

def train(args, agent_opts, train_opts):
    duckie.logger.disabled = True # Disable log messages from ducki  
    env = DuckietownEnv(
        seed = None,
        map_name = "zigzag_dists",
        max_steps = 500001,
        draw_curve = False,
        draw_bbox = False,
        domain_rand = False,
        randomize_maps_on_reset = False,
        accept_start_angle_deg = 4,
        full_transparency = True,
        user_tile_start = None,
        num_tris_distractors = 12,
        enable_leds = False,
    )

    env = ResizeWrapper(env, 80, 80)
    env = SwapDimensionsWrapper(env)
    env = ImageNormalizeWrapper(env)
    env = GymEnvironment(env)

    state_size = env.gym_env.observation_space.shape[0]
    act_size = env.gym_env.action_space.shape[0]
    action_def = ContinuousDefinition(env.gym_env.action_space.shape, \
        env.gym_env.action_space.high, \
        env.gym_env.action_space.low)

    head_network = HeadNet(3)
    head_out_size = 1024
    value_net = ValueNet(head_network, head_out_size)
    target_value_net = ValueNet(head_network, head_out_size)
    actor_net = ActorNet(head_network, head_out_size, action_def, 1e-6)
    critic_net_1 = CriticNet(head_network, head_out_size, act_size)
    critic_net_2 = CriticNet(head_network, head_out_size, act_size)

    multihead_net = SACNetwork(3, act_size)
   
    agent = SACAgent(multihead_net, actor_net, critic_net_1, critic_net_2, value_net, target_value_net, action_def, agent_opts)
    
    if args.checkpoint_path is not None:
        agent.load_model(args.checkpoint_path)
    time = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    trainer = trn.Trainer(agent, env, train_opts)
    trainer.train()



parser = argparse.ArgumentParser()
parser.add_argument("--n-episodes", type=int, help="Number of episodes to train", default=100)
parser.add_argument("--n-iterations", type=int, default=-1, help="Number of max iterations. For unlimited iteration, set it to -1")
parser.add_argument("--replay-buffer-size", type=int, default=100000, help="Size of the replay buffer")
parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
parser.add_argument("--l-rate", type=float, default=0.0003, help="Learning rate for the network")
parser.add_argument("--use-gpu", action="store_true", default=False, help="Set to true to use gpu")
parser.add_argument("--tau", type=float, default=0.005, help="Used to update target network with polyak update")
parser.add_argument("--save-path", type=str, default="duckie_models/simple", help="Path to save the model")
parser.add_argument("--save-freq", type=int, default=2000, help="Number of iterations to save the model")
parser.add_argument("--checkpoint-path", type=str, default=None, help="Path to checkpoint. Use it to resume training")

args = parser.parse_args()

opts = SACAgentOptions()

opts.exp_buffer_size =args.replay_buffer_size
opts.learning_rate = args.l_rate
opts.exp_batch_size = args.batch_size
opts.tau = args.tau
opts.use_gpu = args.use_gpu
opts.clustering = False
opts.save_frequency = args.save_freq
opts.render = False

train_opts = trn.TrainOpts()
train_opts.n_epochs = 1
train_opts.n_episodes = args.n_episodes
train_opts.n_iterations = args.n_iterations # run for 100k iterations
train_opts.save_path = args.save_path

train(args, opts, train_opts)