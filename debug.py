from Agents.ExperienceBuffer import ExperienceBuffer

exp = ExperienceBuffer(5)

exp.add_experience(0,1,4,4,0)
exp.add_experience(0,2,4,5,0)
exp.add_experience(0,3,4,6,0)
exp.add_experience(0,4,4,7,1)

states, actions, rewards, next_states, done = exp.sample(2)
for i in range(2):
    print("Sampled exp: {}, {}, {}, {}, {}".format(states[i], actions[i], rewards[i], next_states[i], done[i]))


import torch
import math

t = torch.Tensor([1,2])
noise = torch.randn(t.shape, dtype=torch.float32)
print(t + noise)