from Environments.Environment import Environment


class GymEnvironment(Environment):
    '''
    Wrapper for gym environments
    '''
    def __init__(self, gym_env):
        self.gym_env = gym_env

    def step(self, action):
        new_state, reward, done, _ = self.gym_env.step(action)
        return new_state, reward, done

    def render(self):
        self.gym_env.render()

    def reset(self):
        return self.gym_env.reset()