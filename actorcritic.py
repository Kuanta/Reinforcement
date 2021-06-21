import gym
import numpy as np
from common import calc_returns
from Agents.Networks import ReinforceNetwork
from Agents.Networks import ActorCriticNetwork
from Agents.ReinforceAgent import ReinforceAgent
from Agents.ActorCriticAgent import ActorCriticAgent
import Trainer as trn

env = gym.make('CartPole-v0')
env._max_episode_steps = 1000
obs = env.reset()

#Agents
acNetwork = ActorCriticNetwork(4, 2)
acAgent = ActorCriticAgent(acNetwork)

reinforceNetwork = ReinforceNetwork(4, 2)
reinforceAgent = ReinforceAgent(reinforceNetwork)

trnOpts = trn.TrainOpts()
trnOpts.n_epochs = 100
trnOpts.n_episodes = 100
trainer = trn.Trainer(agent=reinforceAgent, env=env, opts=trnOpts)
trainer.train()
trainer.test()
acAgent.save_model("ReinforceAgent_1")


