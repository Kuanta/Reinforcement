from Agents.DDPGAgent import DDPGAGent, DDPGAgentOptions
from Environments.Environment import ContinuousDefinition
from Environments.FilterEnv import FilterEnv
from Agents.Networks import ActorNetwork, CriticNetwork
import Trainer as trn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Training Parameters

TRAIN = False
RESUME_TRAINING = False
TEST = True

MODEL_PATH = "models/rl/fitering_varying"

SYSTEM = 0

if SYSTEM == 0:
    # Load data
    system_data = pd.read_csv("data/first_order.csv", header=None).values
    inputs = system_data[:, 1]
    outputs = system_data[:, 2]
    ramp = np.linspace(start=-9.98, stop=9.98, num=500)

elif SYSTEM == 1:
    # A First Order System (1/(s+1))
    inputs = np.ones(500)
    outputs = np.zeros(500)

    for i in range(2, 500):
        outputs[i] = outputs[i-1]*0.9048 + inputs[i]*0.09516


env = FilterEnv(ramp, outputs)
env.reward_weight = 0.001
obs_size = 1
act_size = 1
actor_network = ActorNetwork(n_states=obs_size, n_actions=act_size, device="cuda:0")
critic_network = CriticNetwork(n_states=obs_size, n_actions=act_size, device="cuda:0")

act_def = ContinuousDefinition(lower_limit=-9.8, upper_limit=9.8, shape=1)
opts = DDPGAgentOptions()
opts.discount = 0.99
opts.critic_learning_rate = 0.0005
opts.actor_learning_rate = 0.0005
opts.noise_var = 0.5*(act_def.upper_limit-act_def.lower_limit)*0.01  # or *0.01
opts.target_update_delay = 10
opts.noise_epsilon = 10000  # When this is lower than 0, no noise will be applied
opts.noise_depsilon = 1  # At each iteration, substract this from noise epsilon
opts.exp_batch_size = 100
opts.exp_buffer_size = 10000
opts.uniform_noise_steps = 10

agent = DDPGAGent(actor_network=actor_network, critic_network=critic_network, act_def=act_def, opts=opts)

trnOpts = trn.TrainOpts()
trnOpts.n_epochs = 1
trnOpts.n_episodes = 100
trainer = trn.Trainer(agent=agent, env=env, opts=trnOpts)


if TRAIN:
    if RESUME_TRAINING:
        agent.load_model(MODEL_PATH)
    trainer.train()
    agent.save_model(MODEL_PATH)

if TEST:
    agent.load_model(MODEL_PATH)
    trainer.test()
    plt.figure()
    plt.plot(env.preds, color="RED")
    plt.plot(env.outputs, color="BLUE")
    plt.title("RL Results")
    plt.show()
    plt.figure()
    plt.plot(env.actions, color="ORANGE")
    plt.plot(ramp, color="GREEN")
    plt.title("RL Actions")
    plt.show()
    mse = np.power(env.preds - env.outputs, 2).mean()
    print(env.inputs)
    print("MSE:%f" % (mse))

