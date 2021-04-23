from Agents.DDPGAgent import DDPGAGent, DDPGAgentOptions
from Environments.GymEnvironment import GymEnvironment
from Environments.Environment import ContinuousDefinition
from Agents.Networks import ActorNetwork, CriticNetwork
import Trainer as trn
import gym


env = GymEnvironment(gym.make('LunarLanderContinuous-v2'))
obs_size = env.gym_env.observation_space.shape[0]
act_size = env.gym_env.action_space.shape[0]
actor_network = ActorNetwork(n_states=obs_size, n_actions=act_size, device="cuda:0")
critic_network = CriticNetwork(n_states=obs_size, n_actions=act_size, device="cuda:0")

opts = DDPGAgentOptions()
opts.act_limit_upper = 2
opts.act_limit_lower = -2
opts.critic_learning_rate = 0.0005
opts.actor_learning_rate = 0.0005
opts.noise_var = 0.1
opts.target_update_delay = 10
opts.noise_epsilon = 1  # When this is lower than 0, no noise will be applied
opts.noise_depsilon = 1/100000  # At each iteration, substract this from noise epsilon
opts.exp_batch_size = 64
opts.exp_buffer_size = 10000

action_def = ContinuousDefinition(env.gym_env.action_space.shape, env.gym_env.action_space.high, env.gym_env.action_space.low)
agent = DDPGAGent(actor_network=actor_network, critic_network=critic_network, act_def=action_def, opts=opts)
#agent.load_model("lunar_agent")
trnOpts = trn.TrainOpts()
trnOpts.n_epochs = 1
trnOpts.n_episodes = 250
trainer = trn.Trainer(agent=agent, env=env, opts=trnOpts)
trainer.train()
for i in range(10):
    trainer.test()
agent.save_model("lunar_agent")
