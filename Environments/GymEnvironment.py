from Environments.Environment import Environment

class GymEnvironment(Environment):
    '''
    Wrapper for gym environments
    '''
    def __init__(self, gym_env, pre_process_func = None):
        self.gym_env = gym_env
        self.pre_process_func = pre_process_func

    def step(self, action):
        new_state, reward, done, info = self.gym_env.step(action)
        if self.pre_process_func is not None:
            new_state = self.pre_process_func(new_state)
        return new_state, reward, done, info

    def render(self):
        self.gym_env.render()

    def reset(self):
        init_state = self.gym_env.reset()
        if self.pre_process_func is not None:
            init_state = self.pre_process_func(init_state)
        return init_state