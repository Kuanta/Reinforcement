from Agents.SAC.SequentialSAC import SequentialSACAgentOptions, SequentialSACAgent, SACAgentOptions
from Environments.GymEnvironment import GymEnvironment
from Environments.Environment import ContinuousDefinition
import Trainer as trn
import gym
import os, datetime
from lunar_networks import LunarNetwork, LunarNetworkLSTM

TEST = False  #Set to true to see a trained agent playing Lunar Lander

env = GymEnvironment(gym.make('LunarLanderContinuous-v2'))
state_size = env.gym_env.observation_space.shape[0]
act_size = env.gym_env.action_space.shape[0]
action_def = ContinuousDefinition(env.gym_env.action_space.shape, \
    env.gym_env.action_space.high, \
    env.gym_env.action_space.low)

opts = SequentialSACAgentOptions()

opts.exp_buffer_size = 100000
opts.learning_rate = 0.0003
opts.exp_batch_size = 64
opts.tau = 0.005
opts.use_gpu = True
opts.clustering = False
opts.cluster_samples = 50000
opts.use_elbow_plot = False
opts.n_clusters = 30
opts.n_episodes_exploring = 250  # Number of episodes that uses clustered explore
opts.n_episodes_exploring_least_acts = 250 #Nubmer of episodes that explores by searching for least used acitions
opts.update_cluster_scale = 5 # When to create new clusters? See classify in ExperienceBuffer
opts.cluster_only_for_buffer = True   # If set to true, clustered exploration will only be used for replay buffer filling
opts.render = False
opts.sequence_length = 8
opts.save_frequency = -1
opts.auto_entropy = True

multihead_net = LunarNetworkLSTM(state_size, 128, action_size=act_size)
agent = SequentialSACAgent(multihead_net, action_def, opts)


if  TEST:
    # Test the agent
    agent.load_model("./sac_model/cluster_2/2021-06-12_19-14-41/")
    agent.opts.render = True
    for e in range(10):  # Play 10 episodes
        curr_state = env.reset()
        while True:
            action = agent.act(curr_state, evaluation=True)
            print(action)
            next_state, reward, done, _ = env.step(action)
            curr_state = next_state
            if done:
                break
            env.render()
else:
    
    trnOpts = trn.TrainOpts()
    trnOpts.n_epochs = 1
    trnOpts.n_episodes = 500
    trnOpts.save_path = "test"

    trainer = trn.Trainer(agent, env, trnOpts)
    trainer.train()