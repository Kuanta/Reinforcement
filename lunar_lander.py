from Agents.DDPGAgent import DDPGAGent, DDPGAgentOptions
from Environments.GymEnvironment import GymEnvironment
from Agents.Networks import ActorNetwork, CriticNetwork
import Trainer as trn
import gym

env = GymEnvironment(gym.make('MountainCarContinuous-v0'))
obs_size = env.gym_env.observation_space.shape[0]
act_size = env.gym_env.action_space.shape[0]
actor_network = ActorNetwork(n_states=obs_size, n_actions=act_size, device="cuda:0")
critic_network = CriticNetwork(n_states=obs_size, n_actions=act_size, device="cuda:0")

opts = DDPGAgentOptions()
opts.act_limit_upper = 1
opts.act_limit_lower = -1
opts.critic_learning_rate = 0.01
opts.actor_learning_rate = 0.01
opts.noise_var = 0.1
opts.target_update_delay = 10
opts.noise_var_decay = 1
agent = DDPGAGent(actor_network=actor_network, critic_network=critic_network, opts=opts)
#agent.load_model("ddpg_agent")
trnOpts = trn.TrainOpts()
trnOpts.n_epochs = 1
trnOpts.n_episodes = 500
trainer = trn.Trainer(agent=agent, env=env, opts=trnOpts)
trainer.train()
#trainer.test()
agent.save_model("ddpg_agent")
