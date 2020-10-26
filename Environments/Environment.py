class Environment:
    def __init__(self):
        raise NotImplementedError

    def get_observation(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError