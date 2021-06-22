from Agents.SAC.SACAgent import SACAgentOptions, SACAgent
from Environments.GymEnvironment import GymEnvironment
from Environments.Environment import ContinuousDefinition
from Environments.wrappers import ResizeWrapper, SwapDimensionsWrapper, ImageNormalizeWrapper
import torch
import gym_duckietown.envs.duckietown_env as duckie
from gym_duckietown.envs.duckietown_env import DuckietownEnv
from duckie_networks import *
import argparse

def test(args):
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
   
    agent = SACAgent(multihead_net, actor_net, critic_net_1, critic_net_2, value_net, target_value_net, action_def)
   
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
                print("Total reward:"+total_reward)
                break


parser = argparse.ArgumentParser()
parser.add_argument("--n-episodes", type=int, help="Number of episodes to play")
parser.add_argument("--model-path", type=str, help="Path to the model")

args = parser.parse_args()
test(args)