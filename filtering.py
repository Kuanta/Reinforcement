from Agents.DDPGAgent import DDPGAGent, DDPGAgentOptions
from Environments.FilterEnv import FilterEnv
from Agents.Networks import ActorNetwork, CriticNetwork
import Trainer as trn
import matplotlib.pyplot as plt
import pandas as pd

TRAIN = True
RESUME_TRAINING = False
TEST = True

MODEL_PATH = "fitering"

# Load data
system_data = pd.read_csv("data/first_order.csv", header=None).values
inputs = system_data[:, 1]
outputs = system_data[:, 2]


env = FilterEnv(inputs, outputs)
obs_size = 4
act_size = 1
actor_network = ActorNetwork(n_states=obs_size, n_actions=act_size, device="cuda:0")
critic_network = CriticNetwork(n_states=obs_size, n_actions=act_size, device="cuda:0")

opts = DDPGAgentOptions()
opts.critic_learning_rate = 0.0005
opts.actor_learning_rate = 0.0005
opts.noise_var = 0.98
opts.target_update_delay = 10
opts.noise_epsilon = 1  # When this is lower than 0, no noise will be applied
opts.noise_depsilon = 1/100000  # At each iteration, substract this from noise epsilon
opts.exp_batch_size = 64
opts.exp_buffer_size = 10000
opts.action_scale = 9.8
agent = DDPGAGent(actor_network=actor_network, critic_network=critic_network, opts=opts)

trnOpts = trn.TrainOpts()
trnOpts.n_epochs = 1
trnOpts.n_episodes = 250
trainer = trn.Trainer(agent=agent, env=env, opts=trnOpts)


if TRAIN:
    if RESUME_TRAINING:
        agent.load_model(MODEL_PATH)
    trainer.train()
    agent.save_model(MODEL_PATH)

if TEST:
    agent.load_model(MODEL_PATH)
    trainer.test()
    plt.plot(env.preds, color= "RED")
    plt.plot(env.outputs, color="BLUE")
    plt.show()

