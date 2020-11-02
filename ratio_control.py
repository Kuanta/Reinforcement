from Agents.DDPGAgent import DDPGAGent, DDPGAgentOptions
from Environments.RatioControlEnv import RatioControlEnvironment
from Agents.Networks import ActorNetwork, CriticNetwork
import Trainer as trn
import matplotlib.pyplot as plt

time = [i*0.1 for i in range(100)]
env = RatioControlEnvironment(time)
obs_size = 2
act_size = 1
actor_network = ActorNetwork(n_states=obs_size, n_actions=act_size, device="cuda:0")
critic_network = CriticNetwork(n_states=obs_size, n_actions=act_size, device="cuda:0")

opts = DDPGAgentOptions()
opts.critic_learning_rate = 0.001
opts.actor_learning_rate = 0.001
opts.noise_var = 0.1
opts.target_update_delay = 10
opts.noise_epsilon = 1  # When this is lower than 0, no noise will be applied
opts.noise_depsilon = 1/50000  # At each iteration, substract this from noise epsilon
opts.exp_batch_size = 500
agent = DDPGAGent(actor_network=actor_network, critic_network=critic_network, opts=opts)
agent.load_model("ratio_control")
trnOpts = trn.TrainOpts()
trnOpts.n_epochs = 1
trnOpts.n_episodes = 500
trainer = trn.Trainer(agent=agent, env=env, opts=trnOpts)
#trainer.train()
#agent.save_model("ratio_control")
trainer.test()
plt.figure(1)
plt.plot(env.y1[0:-1])
plt.plot(env.y2[0:-1])
plt.show()
plt.figure(2)
plt.plot(env.r2)
plt.show()