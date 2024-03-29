from Agents.SAC.SACAgent import SACAgent
from Environments.GymEnvironment import GymEnvironment
from Environments.Environment import ContinuousDefinition
from Environments.wrappers import ResizeWrapper, SwapDimensionsWrapper, ImageNormalizeWrapper, EncoderWrapper, TorchifyWrapper
import torch
import gym_duckietown.envs.duckietown_env as duckie
from gym_duckietown.envs.duckietown_env import DuckietownEnv
from duckie_networks import *
import argparse
from Encoder import BetaVAE_B, BetaVAE_H

def test(args):
    duckie.logger.disabled = True # Disable log messages from ducki  
    env = DuckietownEnv(
        seed = None,
        map_name = "4way_bordered",
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

     # Load Encoder
    encoder = BetaVAE_H(10, 3)
    loaded_model = torch.load(args.encoder_path)
    encoder.load_state_dict(loaded_model['model_states']['net'])
    env = ResizeWrapper(env, 64, 64)
    env = SwapDimensionsWrapper(env)
    env = ImageNormalizeWrapper(env)
    env = TorchifyWrapper(env)
    env = EncoderWrapper(env, encoder)
    #env = ActionWrapper(env)
    env = GymEnvironment(env)

    state_size = 14
    act_size = env.gym_env.action_space.shape[0]
    action_def = ContinuousDefinition(env.gym_env.action_space.shape, \
        env.gym_env.action_space.high, \
        env.gym_env.action_space.low)



    multihead_net = DuckieNetwork(state_size, act_size)
   
    agent = SACAgent(multihead_net, action_def)
   
    agent.load_model(args.model_path)
   

    for i in range(args.n_episodes):
        total_reward = 0
        state = env.reset()
        state = torch.from_numpy(state).float().unsqueeze(0)
        while True:
            action = agent.act(state, evaluation=True)
            print(action)
            next_state, reward, done, _ = env.step(action)
            
            state = torch.from_numpy(next_state).float().unsqueeze(0)
            total_reward += reward
            env.render()
            if done:
                print("Total reward:{}".format(total_reward))
                break


parser = argparse.ArgumentParser()
parser.add_argument("--n-episodes", type=int, default=10, help="Number of episodes to play")
parser.add_argument("--model-path", default="duckie_models/simple2", type=str, help="Path to the model")

args = parser.parse_args()
test(args)