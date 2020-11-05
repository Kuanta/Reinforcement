class Environment:
    def step(self, action):
        raise NotImplementedError

    def render(self):
        pass

    def reset(self):
        raise NotImplementedError


class ContinuousDefinition:
    def __init__(self, shape, upper_limit=None, lower_limit=None):
        self.shape = shape
        self.continuous = True
        self.upper_limit = upper_limit
        self.lower_limit = lower_limit

    def __len__(self):
        return self.shape


class DiscreteDefinition:
    def __init__(self, samples):
        self.continuous = False
        self.samples = samples

    def __len__(self):
        return len(self.samples)
