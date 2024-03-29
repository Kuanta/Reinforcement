class BaseAgentOptions:
    '''
    Implements common options that can be used for different agents
    '''
    def __init__(self):
        self.episodic = True
        self.n_episodes = 100
        self.max_steps = 1000  # Max number of steps for an episode
        self.exp_batch_size = 64
        self.exp_buffer_size = 10000
        self.max_epsilon = 1.0
        self.min_epsilon = 0.1
        self.epsilon_decay = 200
        self.use_gpu = False
        self.discount = 0.99
        self.learning_rate = 0.0001
        self.verbose = True
        self.verbose_frequency = 1000
        self.render = True
        self.save_frequency = 1000  # Save frequency in terms of steps
        self.model_path = "model"
