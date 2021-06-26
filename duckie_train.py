from torch.serialization import save
from Agents.SAC.SACAgent import SACAgentOptions, SACAgent
from Environments.GymEnvironment import GymEnvironment
from Environments.Environment import ContinuousDefinition
from Environments.wrappers import DtRewardWrapper, ResizeWrapper, SwapDimensionsWrapper, ImageNormalizeWrapper, EncoderWrapper, ActionWrapper, TorchifyWrapper
import Trainer as trn
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym_duckietown.envs.duckietown_env as duckie
from gym_duckietown.envs.duckietown_env import DuckietownEnv
from gym_duckietown.envs import DuckietownEnv
import os, datetime, argparse
from duckie_networks import *
from Encoder import BetaVAE_B, BetaVAE_H

def train(args, agent_opts, train_opts, rewards=None):
    duckie.logger.disabled = True # Disable log messages from ducki  
    env = DuckietownEnv(
        seed = None,
        map_name = "4way_bordered",
        max_steps = 500000,
        draw_curve = False,
        draw_bbox = False,
        domain_rand = False,
        randomize_maps_on_reset = False,
        accept_start_angle_deg = 4,
        full_transparency = False,
        user_tile_start = None,
        num_tris_distractors = 12,
        enable_leds = False,
        navigation=True,
        num_past_navdirs = 3,
        num_past_positions = 3,
        num_past_actions = 2,
    )

    # Load Encoder
    encoder = BetaVAE_H(10, 3)
    loaded_model = torch.load(args.encoder_path)
    encoder.load_state_dict(loaded_model['model_states']['net'])
    env = ResizeWrapper(env, 64, 64)
    env = SwapDimensionsWrapper(env)
    env = ImageNormalizeWrapper(env)
    env = TorchifyWrapper(env, agent_opts.use_gpu)
    env = EncoderWrapper(env, encoder, agent_opts.use_gpu)
    #env = DtRewardWrapper(env)
    env = ActionWrapper(env)
    env = GymEnvironment(env)

    state_size = 31  # Bottleneck of VAE plus the additional informations
    act_size = env.gym_env.action_space.shape[0]
    action_def = ContinuousDefinition(env.gym_env.action_space.shape, \
        env.gym_env.action_space.high, \
        env.gym_env.action_space.low)

    multihead_net = DuckieNetwork(state_size, act_size)
   
    agent = SACAgent(multihead_net, action_def, agent_opts)

    trainer = trn.Trainer(agent, env, train_opts)
    trainer.train()



parser = argparse.ArgumentParser()
parser.add_argument("--n-episodes", type=int, help="Number of episodes to train", default=5000)
parser.add_argument("--n-iterations", type=int, default=1e6, help="Number of max iterations. For unlimited iteration, set it to -1")
parser.add_argument("--replay-buffer-size", type=int, default=60000, help="Size of the replay buffer")
parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
parser.add_argument("--l-rate", type=float, default=0.0002, help="Learning rate for the network")
parser.add_argument("--use-gpu", action="store_true", default=True, help="Set to true to use gpu")
parser.add_argument("--tau", type=float, default=0.005, help="Used to update target network with polyak update")
parser.add_argument("--save-path", type=str, default="duckie_models/navigation", help="Path to save the model")
parser.add_argument("--save-freq", type=int, default=10000, help="Number of iterations to save the model")
parser.add_argument("--checkpoint-path", type=str, default="duckie_models/navigation/20210626-112148/multihead", help="Path to checkpoint. Use it to resume training")
parser.add_argument("--entropy-scale", type=float, default=0.2, help="Entropy scale used in loss functions")
parser.add_argument("--encoder-path", type=str, default="encoder_model/last", help="Path to the saved encoder model")

args = parser.parse_args()

opts = SACAgentOptions()

opts.exp_buffer_size =args.replay_buffer_size
opts.learning_rate = args.l_rate
opts.exp_batch_size = args.batch_size
opts.tau = args.tau
opts.use_gpu = args.use_gpu
opts.clustering = False
opts.save_frequency = args.save_freq
opts.render = True
opts.entropy_scale = args.entropy_scale

train_opts = trn.TrainOpts()
train_opts.n_epochs = 1
train_opts.n_episodes = args.n_episodes
train_opts.n_iterations = args.n_iterations # run for 100k iterations
train_opts.save_path = args.save_path
train_opts.checkpoint = args.checkpoint_path

train(args, opts, train_opts)