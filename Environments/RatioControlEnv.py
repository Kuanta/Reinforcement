from Environments.Environment import Environment
import numpy as np


class RatioControlEnvironment(Environment):
    def __init__(self, time):
        self.r1 = np.zeros(len(time))
        self.r2 = np.zeros(len(time))
        self.y1 = np.zeros(len(time))
        self.y2 = np.zeros(len(time))
        self.curr_t = 0
        self.time = time

        # System parameters
        self.system_params = [0.9048, 0.09516, 0.9048, 0.1903]

    def step(self, action):
        self.update_y2(action)
        reward = -np.power(np.array((self.y1[self.curr_t]-self.y2[self.curr_t])), 2)
        self.curr_t = self.curr_t + 1
        done = False
        if self.curr_t+1 >= len(self.time):
            done = True
        obs = self.get_curr_observation()
        return obs, reward, done

    def render(self):
        pass

    def reset(self):
        self.r1 = np.ones(len(self.time))
        self.r2 = np.zeros(len(self.time))
        self.y1 = np.zeros(len(self.time))
        self.y2 = np.zeros(len(self.time))
        self.curr_t = 1
        self.update_y1()
        init_obs = self.get_curr_observation()
        return init_obs

    def update_y1(self):
        self.y1[self.curr_t] = self.y1[self.curr_t-1]*self.system_params[0] + self.r1[self.curr_t-1]*self.system_params[1]

    def update_y2(self, action):
        self.r2[self.curr_t] = action
        self.y2[self.curr_t] = self.y2[self.curr_t-1]*self.system_params[2] + self.r2[self.curr_t]*self.system_params[3]

    def get_curr_observation(self):
        self.update_y1()
        return np.array((self.y1[self.curr_t], self.y1[self.curr_t-1]))

    def simulate(self, agent):
        curr_state = self.reset()
        actions = []
        for i in range(len(self.time)):
            action = agent.act(curr_state)
            next_state, reward, _ = self.step(action)
            curr_state = next_state
