class BaseAgentOptions:
    '''
    Implements common options that can be used for different agents
    '''
    def __init__(self):
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
